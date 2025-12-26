#include <ozaki/cpu/ozi.hpp>

namespace ozicpu {

void matmul(int32_t *__restrict C, const int8_t *__restrict A, const int8_t *__restrict B, size_t m, size_t n, size_t k) {
  for (size_t mm = 0; mm < m; ++mm) {
    for (size_t nn = 0; nn < n; ++nn) {
      int32_t res = 0;
      for (size_t kk = 0; kk < k; ++kk) {
        res += static_cast<int32_t>(A[mm + m * kk]) * static_cast<int32_t>(B[kk + nn * k]);
      }
      C[mm + m * nn] = res;
    }
  }
}

}
