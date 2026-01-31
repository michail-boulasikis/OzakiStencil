#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <quadmath.h>

#include "../include/ozaki/utils/tensor.hpp"

void generate_gaussian_peaks_fp128(const std::string &output_path, size_t n, size_t n_peaks, bool halo_avg) {
    Tensor<__float128> tensor;
    tensor.shape = {n, n};
    tensor.data.resize(n * n);
    std::mt19937_64 rng(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<double> peak_dist(0.3, 0.7);
    std::uniform_real_distribution<double> std_dist(0.001, 0.01);

    // Generate peaks
    std::vector<__float128> peak_x(n_peaks), peak_y(n_peaks), peak_std(n_peaks);
    for (size_t p = 0; p < n_peaks; ++p) {
        peak_x[p] = static_cast<__float128>(peak_dist(rng));
        peak_y[p] = static_cast<__float128>(peak_dist(rng));
        peak_std[p] = static_cast<__float128>(std_dist(rng));
    }

    // Compute grid values
    __float128 min_abs = 1e100Q;
    __float128 max_abs = 0.0Q;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            __float128 x = static_cast<__float128>(i) / static_cast<__float128>(n - 1);
            __float128 y = static_cast<__float128>(j) / static_cast<__float128>(n - 1);

            __float128 value = 0.0Q;
            for (size_t p = 0; p < n_peaks; ++p) {
                __float128 dx = x - peak_x[p];
                __float128 dy = y - peak_y[p];
                __float128 dist_sq = dx * dx + dy * dy;
                value += expq(-dist_sq / peak_std[p]);
            }

            tensor.data[i * n + j] = value;
            __float128 abs_val = fabsq(value);
            if (abs_val > 0.0Q) {
                min_abs = fminq(min_abs, abs_val);
            }
            max_abs = fmaxq(max_abs, abs_val);
        }
    }

    // Apply halo averaging if requested
    if (halo_avg) {
        __float128 sum = 0.0Q;
        for (const auto &val : tensor.data) {
            sum += val;
        }
        __float128 avg = sum / static_cast<__float128>(tensor.data.size());

        for (size_t i = 0; i < n; ++i) {
            tensor.data[i] = avg;           // Top edge
            tensor.data[(n-1) * n + i] = avg; // Bottom edge
            tensor.data[i * n] = avg;         // Left edge
            tensor.data[i * n + (n-1)] = avg; // Right edge
        }
    }

    write_tensor(output_path, tensor);
    std::cout << "Written " << n << "x" << n << " fp128 tensor to " << output_path << std::endl;

    char buf_min[64], buf_max[64];
    quadmath_snprintf(buf_min, sizeof(buf_min), "%.6Qe", min_abs);
    quadmath_snprintf(buf_max, sizeof(buf_max), "%.6Qe", max_abs);
    std::cout << "\tTensor |min| = " << buf_min << std::endl;
    std::cout << "\tTensor |max| = " << buf_max << std::endl;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " OUTPUT_FILE [--N SIZE] [--halo-avg]" << std::endl;
        return 1;
    }

    std::string output_path = argv[1];
    size_t n = 64;
    bool halo_avg = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--N" && i + 1 < argc) {
            n = std::stoull(argv[++i]);
        } else if (arg == "--halo-avg") {
            halo_avg = true;
        }
    }

    generate_gaussian_peaks_fp128(output_path, n, 10, halo_avg);
    return 0;
}
