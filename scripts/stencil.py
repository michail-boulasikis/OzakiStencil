import numpy as np
import cuda.bindings.nvrtc as nvrtc
import cuda.bindings.driver as driver
import ctypes
import os
import sys
from typing import Tuple, List, Optional
import textwrap
import numpy as np

DTYPE_MAP = {
    "double": np.float64,
    "float": np.float32,
    "int": np.int32,
    "long": np.int64,
    "short": np.int16,
    "int8_t": np.int8,
    "uint8_t": np.uint8,
    "int16_t": np.int16,
    "uint16_t": np.uint16,
    "int32_t": np.int32,
    "uint32_t": np.uint32,
    "int64_t": np.int64,
    "uint64_t": np.uint64,
}
# Duplicates are not a problem, for example int is the same as int32_t, so I don't mind whichever comes on top
DTYPE_MAP_INV = {np.dtype(v): k for k, v in DTYPE_MAP.items()}


def strides_from_shape(shape: Tuple[int, ...]):
    tmp_strides = list(shape[::-1])
    tmp_strides[1:] = list(np.cumprod(tmp_strides[:-1]))
    tmp_strides[0] = 1
    return tuple(tmp_strides)


def shape_from_strides(strides: Tuple[int, ...], n_elems: int):
    sorted_indices = strides[::-1]
    shape = [0] * len(strides)
    for i, current_idx in enumerate(sorted_indices):
        current_stride = strides[current_idx]
        next_val = strides[sorted_indices[i + 1]] if i + 1 < len(strides) else n_elems
        shape[current_idx] = next_val // current_stride
    return tuple(shape)


def linear_from_multi(index, strides):
    return int(np.sum(index * np.array(strides)))


def multi_from_linear(index, strides, n_elems):
    shape = shape_from_strides(strides, n_elems)
    return tuple([(x / s) % dim for x, s, dim in zip(index, strides, shape)])


class Stencil(object):
    def __init__(self, coeffs: np.ndarray, origin: np.ndarray):
        assert len(coeffs.shape) == origin.shape[0]
        self.coeffs = coeffs
        self.origin = origin.astype(np.int32)

    @staticmethod
    def from_file(fname: str):
        if not os.path.exists(fname):
            raise FileNotFoundError(f"File not found: {fname}")
        with open(fname, "r") as f:
            tokens = f.read().split()
        if not tokens:
            raise ValueError("File is empty")
        iterator = iter(tokens)

        try:
            # 1. Parse Type
            type_str = next(iterator)
            if type_str not in DTYPE_MAP:
                raise ValueError(f"Unsupported or unknown type: {type_str}")
            dtype = DTYPE_MAP[type_str]
            # 2. Parse Number of Dimensions
            ndims = int(next(iterator))
            # 3. Parse Dimensions (Shape)
            shape = []
            for _ in range(ndims):
                shape.append(int(next(iterator)))
            # 4. Parse Origin
            origin = []
            for _ in range(ndims):
                origin.append(int(next(iterator)))
            # 5. Parse Coefficients
            data_tokens = list(iterator)
            # Check if data length matches the shape volume
            expected_count = np.prod(shape)
            if len(data_tokens) != expected_count:
                raise ValueError(
                    f"Expected {expected_count} coefficients, found {len(data_tokens)}"
                )
            coeffs = np.array(data_tokens, dtype=dtype).reshape(shape)
            return cls(coeffs, np.array(origin))
        except StopIteration:
            raise ValueError("Unexpected end of file while parsing header info.")
        except ValueError as e:
            raise ValueError(f"Error parsing file content: {e}")


class StencilKernelGenerator(object):
    def __init__(
        self,
        stencil: Stencil,
        input_shape: Tuple[int, ...],
        kernel_name: str = "stencil",
    ):
        self.stencil = stencil
        self.shape = input_shape
        self.ndim = len(input_shape)
        self.kernel_name = kernel_name

        # Mapping dimension index to variable names
        self.var_names = "ijk"[: self.ndim]
        self.cuda_dims = "xyz"[: self.ndim]

        self._validate()

    def _validate(self):
        if self.ndim > 3:
            raise NotImplementedError(
                f"Stencils > 3D are not supported (Got {self.ndim})."
            )
        if self.stencil.coeffs.dtype not in DTYPE_MAP_INV:
            raise ValueError(f"Unsupported dtype: {self.stencil.coeffs.dtype}")

    @property
    def c_type(self) -> str:
        return DTYPE_MAP_INV[self.stencil.coeffs.dtype]

    def _generate_indices(self) -> str:
        lines = []
        for v, c in zip(self.var_names, self.cuda_dims):
            lines.append(
                f"const int {v} = threadIdx.{c} + blockIdx.{c} * blockDim.{c};"
            )
        return "\n".join(lines)

    def _generate_bounds_check(self) -> str:
        checks = []
        for i, (var, limit) in enumerate(zip(self.var_names, self.shape)):
            start = self.stencil.origin[i]
            end = limit - self.stencil.origin[i]
            checks.append(f"({var} >= {start} && {var} < {end})")

        condition = " && ".join(checks)
        return f"if (!({condition})) return;"

    def _get_linear_index(
        self, offsets: Optional[Tuple[int, ...]] = None, shift_by_origin=True
    ) -> str:
        if offsets is None:
            offsets = (0,) * self.ndim

        # Adjust offsets relative to the stencil origin if needed
        if shift_by_origin:
            adjusted_offsets = [
                o - origin for o, origin in zip(offsets, self.stencil.origin)
            ]
        else:
            adjusted_offsets = offsets

        components = []
        stride = 1

        for var, dim, off in zip(self.var_names, self.shape, adjusted_offsets):
            term = (
                f"{var}"
                if off == 0
                else (f"({var} + {off})" if off > 0 else f"({var} - {-off})")
            )
            # Optimization: Don't multiply by 1
            components.append(term if stride == 1 else f"({term}) * {stride}")
            stride *= dim

        return " + ".join(components)

    def _generate_update_expression(self) -> str:
        terms = []

        for index_tuple, coeff in np.ndenumerate(self.stencil.coeffs):
            if coeff == 0:
                continue

            coeff_str = f"{coeff}"
            if "float" in self.c_type and "." not in coeff_str:
                coeff_str += ".0"

            neighbor_idx = self._get_linear_index(offsets=index_tuple)
            terms.append(f"in[{neighbor_idx}] * ({coeff_str})")

        return " + ".join(terms)

    def generate_source(self) -> str:
        code = f"""
        extern "C" __global__ void {self.kernel_name}({self.c_type} *__restrict__ out, const {self.c_type} *in) {{
            // 1. Thread Indexing
            {self._generate_indices()}

            // 2. Boundary Check
            {self._generate_bounds_check()}

            // 3. Stencil Computation
            // Target Index (Center)
            const int current_idx = {self._get_linear_index(shift_by_origin=False)};
            
            // Weighted Sum
            out[current_idx] = {self._generate_update_expression()};
        }}
        """
        return textwrap.dedent(code).strip()


# --- Public API Wrapper ---
def ptx_kernel_from_stencil(
    stencil: Stencil, input_array_shape: Tuple[int, ...], kernel_name="stencil"
) -> str:
    generator = StencilKernelGenerator(
        stencil, input_array_shape, kernel_name=kernel_name
    )
    return generator.generate_source()


# --- Helper to check CUDA Errors ---
def check_cuda_err(err, msg):
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA Error: {err} - {msg}")


# --- Execution Engine ---
def run_benchmark(
    ptx_code: bytes,
    kernel_name: str,
    input_shape: Tuple[int, ...],
    dtype: type,
    warmup_steps: int = 10,
    profiled_steps: int = 128,
):
    # Cuda init stuff
    (err,) = driver.cuInit(0)
    check_cuda_err(err, "cuInit")

    err, device = driver.cuDeviceGet(0)
    check_cuda_err(err, "cuDeviceGet")

    err, ctx = driver.cuCtxCreate(None, 0, device)
    check_cuda_err(err, "cuCtxCreate")

    try:
        # loading the kernel
        err, module = driver.cuModuleLoadData(ptx_code)
        check_cuda_err(err, "cuModuleLoadData")

        err, kernel = driver.cuModuleGetFunction(module, kernel_name.encode("utf-8"))
        check_cuda_err(err, "cuModuleGetFunction")

        err, register_spills = driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel
        )
        check_cuda_err(err, "cuFuncGetAttribute")
        print(f"REG_SPILL={register_spills}")

        # calculating sizes
        itemsize = np.dtype(dtype).itemsize
        num_elements = np.prod(input_shape)
        size_bytes = num_elements * itemsize

        h_in = np.random.randn(*input_shape).astype(dtype)

        err, d_in = driver.cuMemAlloc(size_bytes)
        check_cuda_err(err, "Alloc in")
        err, d_out = driver.cuMemAlloc(size_bytes)
        check_cuda_err(err, "Alloc out")

        # copy HtoD
        (err,) = driver.cuMemcpyHtoD(d_in, h_in, size_bytes)
        check_cuda_err(err, "Memcpy HtoD")

        # launch config
        block_dim = (32, 32, 1)
        grid_dim = (
            (input_shape[1] + block_dim[1] - 1) // block_dim[1],
            (input_shape[0] + block_dim[0] - 1) // block_dim[0],
            1,
        )

        c_d_out = ctypes.c_uint64(int(d_out))
        c_d_in = ctypes.c_uint64(int(d_in))

        # kernel params: void* args[] = { &out, &in };
        args_array = (ctypes.c_void_p * 2)(
            ctypes.addressof(c_d_out), ctypes.addressof(c_d_in)
        )

        print(f"Launching {kernel_name} | Grid: {grid_dim} | Block: {block_dim}")

        # warmup launch
        for _ in range(warmup_steps):
            driver.cuLaunchKernel(kernel, *grid_dim, *block_dim, 0, 0, args_array, 0)
            args_array[0], args_array[1] = args_array[1], args_array[0]

        # profiled launch
        for _ in range(profiled_steps):
            driver.cuLaunchKernel(kernel, *grid_dim, *block_dim, 0, 0, args_array, 0)
            args_array[0], args_array[1] = args_array[1], args_array[0]

        (err,) = driver.cuCtxSynchronize()
        check_cuda_err(err, "Sync")
        print("Kernel execution finished successfully.")

    finally:
        if "d_in" in locals():
            driver.cuMemFree(d_in)
        if "d_out" in locals():
            driver.cuMemFree(d_out)
        driver.cuCtxDestroy(ctx)
