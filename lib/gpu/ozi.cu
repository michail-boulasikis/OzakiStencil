#include "ozaki/gpu/gpu_checks.cuh"
#include <climits>
#include <cstdint>
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <mma.h>
#include <ozaki/gpu/ozi.cuh>
#include <vector_types.h>

using namespace nvcuda;

#define TILE_DIM 16
using FragA = wmma::fragment<wmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, int8_t, wmma::col_major>;
using FragB = wmma::fragment<wmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, int8_t, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, int32_t>;

CUDA_DEVICE CUDA_FORCEINLINE void shifted_accumulation(int32_t *acc, FragC &partial, FragC &temp,
                                                       const int32_t byte_shifts, const size_t lda) {
  wmma::load_matrix_sync(temp, acc, TILE_DIM, wmma::mem_col_major);
  for (int32_t j = 0; j < temp.num_elements; ++j) {
    temp.x[j] += partial.x[j] << (byte_shifts * CHAR_BIT);
  }
  wmma::store_matrix_sync(acc, temp, lda, wmma::mem_col_major);
}

CUDA_DEVICE CUDA_FORCEINLINE void byte_permute_slice(int4 *bytes) {
  int32_t t0 = __byte_perm(bytes->x, bytes->y, 0x5140);
  int32_t t1 = __byte_perm(bytes->x, bytes->y, 0x7362);
  int32_t t2 = __byte_perm(bytes->z, bytes->w, 0x5140);
  int32_t t3 = __byte_perm(bytes->z, bytes->w, 0x7362);
  bytes->w = __byte_perm(t0, t2, 0x6420);
  bytes->z = __byte_perm(t0, t2, 0x7531);
  bytes->y = __byte_perm(t1, t3, 0x6420);
  bytes->x = __byte_perm(t1, t3, 0x7531);
}

CUDA_GLOBAL void ozi_mul_stencil(int8_t *__restrict__ out, const int8_t *__restrict__ stencil,
                                 const int8_t *__restrict__ data, const int32_t num_slices, const size_t m,
                                 const size_t n, const size_t halo_size) {

  constexpr size_t TILE_PADDING = 4; // load_matrix_sync is picky about alignment
  const size_t NUM_WARPS = blockDim.x / WARP_SIZE;

  // the intermediate shared memory step is needed due to alignment issues
  // load_matrix_sync requires a 256-bit aligned address, which is impossible to achieve due to the halo_size
  // instead, I manually copy into a shmem region which is aligned
  // no padding here because load_matrix_sync is picky about stride/leading dimensions
  // memroy size = NUM_WARPS * TILE_SIZE * TILE_SIZE
  extern __shared__ __align__(32) int8_t tile[];
  // we also need an accumulator
  // padding is important here since I expect this accumulator to go back and forth between shmem and registers often
  // so shared memory bank conflicts will be very costly
  __shared__ __align__(32) int32_t accumulator[(TILE_DIM + TILE_PADDING) * TILE_DIM];
  // finally, buffers to hold the stencil
  __shared__ __align__(32) int8_t shared_stencil[2 * TILE_DIM * TILE_DIM];

  // step 1: load the stencil in a fragment, this is needed for all warps and remains constant
  // I also declare all fragments here
  FragA stencil_frag;
  FragB data_frag;
  FragC acc_frag, temp_frag;
  wmma::load_matrix_sync(stencil_frag, stencil, TILE_DIM);

  // step 3: preparing indexing variables
  const int32_t WARP_ID = threadIdx.x / WARP_SIZE;
  const int32_t LANE_ID = threadIdx.x % WARP_SIZE;
  const size_t block_x = blockIdx.x * (TILE_DIM - halo_size);
  const size_t block_y = blockIdx.y * (TILE_DIM - halo_size);
  int32_t current_slice = num_slices - WARP_ID - 1;

  // step 4: matrix multiplication and accumulation over the slices
  while (current_slice >= 0) {
    // step 4.1 prepare accumulator
    wmma::fill_fragment(acc_frag, 0);
    // step 4.2: load slice
    const size_t slice_block_base = block_x + block_y * m + current_slice * m * n;
    const int *src_row_ptr = reinterpret_cast<const int *>(data + slice_block_base);
    int *dst_int_ptr = reinterpret_cast<int *>(tile + WARP_ID * (TILE_DIM * TILE_DIM));
// Each lane loads 2 integers (8 bytes)
#pragma unroll
    for (int32_t k = 0; k < 2; ++k) {
      int32_t element_idx = LANE_ID + k * WARP_SIZE; // 0..63 integers
      int32_t row = element_idx / 4;                 // 4 ints per row (16 bytes)
      int32_t col_int = element_idx % 4;

      // pointer arithmetic: src_row_ptr + row * (m/sizeof(int)) + col_int
      // NOTE: If m is not divisible by 4, this math breaks. Assuming m is aligned.
      if (row < TILE_DIM) {
        dst_int_ptr[element_idx] = src_row_ptr[row * (m >> 2) + col_int];
      }
    }

    // step 4.3 load tile from shmem and compute matrix multiplication
    wmma::load_matrix_sync(data_frag, tile + WARP_ID * (TILE_DIM * TILE_DIM), TILE_DIM);
    wmma::mma_sync(acc_frag, stencil_frag, data_frag, acc_frag);
    // step 4.4: the tricky part, accumulation
    // step 4.4.1: if these are the LSBs, we throw away the part that can be ignored when maintaining precision
    // and we initialize the accumulator
    if (current_slice == num_slices - 1) {
      for (int32_t j = 0; j < acc_frag.num_elements; ++j)
        acc_frag.x[j] /= 256;
      wmma::store_matrix_sync(accumulator, acc_frag, TILE_DIM, wmma::mem_col_major);
    }
    __syncthreads();
    // at this point the accumulator has 8 bits of information on the LSBs of each element
    // (no matter the loop iteration)

    // step 4.4.2: each warp accumulates its partial result shifted
    // (serialized, unfortunately - I could use two accumulators and speed this up in some cases)
    // (but I first want to see if I am memory bound, in which case I can afford to wait)
    if (current_slice != num_slices - 1) {
      if (WARP_ID == 0)
        shifted_accumulation(accumulator, acc_frag, temp_frag, 0, TILE_DIM);
      __syncthreads(); // is this a deadlock? (__syncthreads() in if statement)
      if (WARP_ID == 1)
        shifted_accumulation(accumulator, acc_frag, temp_frag, 0, TILE_DIM);
      __syncthreads();
      if (WARP_ID == 2)
        shifted_accumulation(accumulator, acc_frag, temp_frag, 1, TILE_DIM);
      __syncthreads();
      if (WARP_ID == 3)
        shifted_accumulation(accumulator, acc_frag, temp_frag, 2, TILE_DIM);
      __syncthreads();
    }
    // step 4.5: store into memory the <4 partial results we got in this loop iteration
    // the results should be in the same layout the data is, so we need to reshape them

    if (threadIdx.x < (TILE_DIM * TILE_DIM) / 4) { // we need to load 4 elements per thread, so 256 / 4 = 64 threads
                                                   // I have enough threads to cover the tile w/ vectorized loads,
                                                   // I do not need to loop
      int32_t row = threadIdx.x % (TILE_DIM / 4);
      int32_t col = threadIdx.x / (TILE_DIM / 4);
      // 1 2 3 4 5 6 7 8 9 a b c d e f 10
      // | | | | | | | | | | | | | | | | (each line is 4 elements tall - col-major)
      // | | | | | | | | | | | | | | | |
      // | | | | | | | | | | | | | | | |
      // | | | | | | | | | | | | | | | |
      // - - - - - - - - - - - - - - - - (padding for bank conflict avoidance)
      int4 loaded_partial = *(reinterpret_cast<int4 *>(accumulator) + row + col * (TILE_DIM + TILE_PADDING) / 4);
      // step 4.5.2: byte permutation
      byte_permute_slice(&loaded_partial);
      // step 4.5.3: store back to shared memory before doing byte-wise store. Handling the halo is easier that way,
      // unfortunately
      *(reinterpret_cast<int4 *>(accumulator) + row + col * (TILE_DIM + TILE_PADDING) / 4) = loaded_partial;
    }
    __syncthreads();

    // we have (16 - halo) * (16 - halo) bytes per warp
    if (LANE_ID > halo_size && LANE_ID < TILE_DIM - halo_size) {
      int8_t *byte_acc = reinterpret_cast<int8_t *>(accumulator) + WARP_ID * (TILE_DIM + TILE_PADDING) * TILE_DIM +
                         LANE_ID * (TILE_DIM + TILE_PADDING);
      //      for (int32_t j = halo_size; j < TILE_DIM - halo_size; ++j)
      //        out[] = byte_acc[j];
    }

    current_slice -= NUM_WARPS;
  }
}
