#ifndef HEADER_OZAKI_GPU_CUH
#define HEADER_OZAKI_GPU_CUH

#ifndef M
#define M 64
#endif
#ifndef N
#define N 256
#endif
#ifndef K
#define K 32
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

struct OzakliSlice {
  int32_t exp;
  int8_t data[N * K];
}

namespace {
  template <size_t NUM_THREADS>
    requires(NUM_THREADS % WARP_SIZE == 0)
  __device__ __forceinline__ void reduce_abs_max_exp(volatile double *workspace, const double *sdata) {

    size_t i = threadIdx.x;

    // Reduction to NUM_THREADS elements (with vectorized load from shmem)
    double local_abs_max = 0.0;
#pragma unroll 8
    while (2 * i < N * K) {
      const double2 local vals = *reinterpret_cast<const double2 *>(sdata + 2 * i);
      local_abs_max = fmax(fmax(local_abs_max, fabs(vals.x)), fabs(vals.y));
      i += NUM_THREADS;
    }
    // Reduction to (NUM_THREADS / 32) elements (max 32 cause NUM_THREADS is at most 1024)
    const uint32_t lane_id = threadIdx.x % WARP_SIZE;
    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    if (lane_id < 16) local_abs_max = fmax(local_abs_max, __shfl_down_sync(0xFFFF'FFFF, local_abs_max, 16);
    if (lane_id < 8)  local_abs_max = fmax(local_abs_max, __shfl_down_sync(0xFFFF'FFFF, local_abs_max, 8);
    if (lane_id < 4)  local_abs_max = fmax(local_abs_max, __shfl_down_sync(0xFFFF'FFFF, local_abs_max, 4);
    if (lane_id < 2)  local_abs_max = fmax(local_abs_max, __shfl_down_sync(0xFFFF'FFFF, local_abs_max, 2);
    if (lane_id < 1)  local_abs_max = fmax(local_abs_max, __shfl_down_sync(0xFFFF'FFFF, local_abs_max, 1);
    if (lane_id == 0) workspace[warp_id] = local_abs_max;
    __syncthreads();

    // <32 elements in workspace, can be dealt with a single warp.
    // warp reduction to 1 element, which is stored to workspace[0]
    if ((warp_id == 0) && (lane_id < (NUM_THREADS / 32))) {
        uint32_t mask = __activemask();
        local_abs_max = workspace[threadIdx.x];
#pragma unroll // compiler is gonna know at compile time how many loop iterations this is.
        for (uint32_t offset = (NUM_THREADS / 64); offset > 0; offset /= 2)
          local_abs_max = fmax(local_abs_max, __shfl_down_sync(mask, local_abs_max, offset));
        if (threadIdx.x == 0) {
          workspace[0] = local_abs_max;
        }
    }
  }

  // Computes the 8-bit approximation (scaled by exp) of double x, and stores the error of this transformation to &sxres
  // as a side effect
  __device__ __forceinline__ int8_t residue(const double x, const int32_t exponent, volatile double &sxres) {
    constexpr auto lo = std::numeric_limits<int8_t>::min() + 1;
    constexpr auto hi = std::numeric_limits<int8_t>::max();
    double y = llround(ldexp(x, exponent));
    // TODO: There's gotta be a better way to do this in GPU. Clamp or w/e.
    if (r < lo)
      r = lo;
    if (r > hi)
      r = hi;
    *sxres = *sxres - ldexp(static_cast<double>(r), -exponent);
    return static_cast<int8_t>(r);
  }
}

// num_threads for this one should be 128 - otherwise it won't work
template <size_t N_SLICES>
__global__ void ozaki_slice(const __grid_constant__ CUtensorMap out,
                            const __grid_constant__ CUtensorMap in) {
  __shared__ alignas(128) double s_tile[2 * K * N];
  __shared__ alignas(128) int8_t s_tile_res[2 * K * N];
  __shared__ double sworkspace[WARP_SIZE];

  double *buf_current = s_tile;
  double *buf_next = s_tile + (K * N);
  int8_t *rbuf_current = s_tile_res;
  int8_t *rbuf_next = s_tile_res + (K * N);

  uint32_t x, y, x_prev, y_prev;
  size_t linear_idx = K * N * blockIdx.x;
  x = K * N * blockIdx.x / out.globalDim[0];
  y = K * N * blockIdx.x / out.globalDim[1];

  load_buffer(buf_current, in, x, y);
  wait_for_curr();

  // loop
  while (linear_idx < out.globalDim[0] * out.globalDim[1])
  {
    x_prev = x;
    y_prev = y;
    load_buffer(buf_next, in, x, y);
    compute(rbuf_current, buf_current, sworkspace);
    store(out, rbuf_current, x_prev, y_prev);
    wait_for_next();

    std::swap(buf_current, buf_next);
    std::swap(rbuf_current, rbuf_next);
  }

  compute(rbuf_current, buf_current, sworkspace);
  store(out, rbuf_current, x, y);

  //  cuda::experimental::cp_async_tensor_2d_global_to_shared(&sslice, &in, copy_row, copy_col, barrier);
}

#endif // HEADER_OZAKI_GPU_CUH
