#include <ozaki/core/oztraits.hpp>
#include <ozaki/utils/tensor.hpp>

#include <boost/program_options.hpp>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace po = boost::program_options;

// ---- __float128 Gaussian helpers -------------------------------------------

static __float128 q_exp_gauss(__float128 dx, __float128 dy, __float128 inv_std) {
    __float128 r2 = dx * dx + dy * dy;
    return expq(-r2 * inv_std);
}

static __float128 q_exp_gauss3(__float128 dx, __float128 dy, __float128 dz, __float128 inv_std) {
    __float128 r2 = dx * dx + dy * dy + dz * dz;
    return expq(-r2 * inv_std);
}

// ---- 2D generator ----------------------------------------------------------

static Tensor<__float128> generate_2d(size_t n, int n_peaks, bool halo_avg, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> center_dist(0.3, 0.7);
    std::uniform_real_distribution<double> std_dist(0.001, 0.01);

    // Build FP128 grid [0, 1]
    std::vector<__float128> grid(n);
    for (size_t i = 0; i < n; ++i)
        grid[i] = (__float128)i / (__float128)(n - 1);

    std::vector<__float128> output(n * n, (__float128)0.0);

    for (int p = 0; p < n_peaks; ++p) {
        __float128 px  = (__float128)center_dist(rng);
        __float128 py  = (__float128)center_dist(rng);
        __float128 std = (__float128)std_dist(rng);
        __float128 inv = (__float128)1.0 / std;

        for (size_t j = 0; j < n; ++j) {        // row (Y)
            __float128 dy = grid[j] - py;
            for (size_t i = 0; i < n; ++i) {    // col (X)
                __float128 dx = grid[i] - px;
                output[j * n + i] += q_exp_gauss(dx, dy, inv);
            }
        }
    }

    if (halo_avg) {
        __float128 sum = (__float128)0.0;
        for (auto v : output) sum += v;
        __float128 avg = sum / (__float128)(n * n);

        for (size_t i = 0; i < n; ++i) {
            output[0 * n + i]       = avg;   // top row
            output[(n - 1) * n + i] = avg;   // bottom row
            output[i * n + 0]       = avg;   // left col
            output[i * n + (n - 1)] = avg;   // right col
        }
    }

    return {{n, n}, std::move(output)};
}

// ---- 3D generator ----------------------------------------------------------

static Tensor<__float128> generate_3d(size_t n, int n_peaks, bool halo_avg, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> center_dist(0.3, 0.7);
    std::uniform_real_distribution<double> std_dist(0.001, 0.01);

    std::vector<__float128> grid(n);
    for (size_t i = 0; i < n; ++i)
        grid[i] = (__float128)i / (__float128)(n - 1);

    std::vector<__float128> output(n * n * n, (__float128)0.0);

    for (int p = 0; p < n_peaks; ++p) {
        __float128 px  = (__float128)center_dist(rng);
        __float128 py  = (__float128)center_dist(rng);
        __float128 pz  = (__float128)center_dist(rng);
        __float128 std = (__float128)std_dist(rng);
        __float128 inv = (__float128)1.0 / std;

        for (size_t k = 0; k < n; ++k) {           // dim 0
            __float128 dz = grid[k] - pz;
            for (size_t j = 0; j < n; ++j) {       // dim 1
                __float128 dy = grid[j] - py;
                for (size_t i = 0; i < n; ++i) {   // dim 2
                    __float128 dx = grid[i] - px;
                    output[k * n * n + j * n + i] += q_exp_gauss3(dx, dy, dz, inv);
                }
            }
        }
    }

    if (halo_avg) {
        __float128 sum = (__float128)0.0;
        for (auto v : output) sum += v;
        __float128 avg = sum / (__float128)(n * n * n);

        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i) {
                output[0 * n * n + j * n + i]       = avg;   // k=0 face
                output[(n-1) * n * n + j * n + i]   = avg;   // k=n-1 face
            }
        for (size_t k = 0; k < n; ++k)
            for (size_t i = 0; i < n; ++i) {
                output[k * n * n + 0 * n + i]       = avg;   // j=0 face
                output[k * n * n + (n-1) * n + i]   = avg;   // j=n-1 face
            }
        for (size_t k = 0; k < n; ++k)
            for (size_t j = 0; j < n; ++j) {
                output[k * n * n + j * n + 0]       = avg;   // i=0 face
                output[k * n * n + j * n + (n-1)]   = avg;   // i=n-1 face
            }
    }

    return {{n, n, n}, std::move(output)};
}

// ---- main ------------------------------------------------------------------

int main(int argc, char *argv[]) {
    std::string output_file;
    size_t      n         = 64;
    int         n_peaks   = 10;
    int         ndim      = 2;
    uint64_t    seed      = 42;
    bool        halo_avg  = false;

    po::options_description desc("gen-tensor: generate FP128 Gaussian-peak tensor files");
    // clang-format off
    desc.add_options()
        ("help,h",   "Produce help message")
        ("output,o", po::value<std::string>(&output_file)->required(),
                     "Path to output binary tensor file")
        ("N",        po::value<size_t>(&n)->default_value(64),
                     "Grid size per dimension")
        ("peaks",    po::value<int>(&n_peaks)->default_value(10),
                     "Number of Gaussian peaks")
        ("dims,d",   po::value<int>(&ndim)->default_value(2),
                     "Dimensionality: 2 or 3")
        ("seed",     po::value<uint64_t>(&seed)->default_value(42),
                     "RNG seed")
        ("halo-avg", po::bool_switch(&halo_avg),
                     "Set boundary cells to the domain mean")
        ;
    // clang-format on

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) { std::cout << desc << "\n"; return 0; }
        po::notify(vm);
    } catch (const po::error &e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << "Use -h or --help to see usage.\n";
        return 1;
    }

    if (ndim != 2 && ndim != 3) {
        std::cerr << "Error: --dims must be 2 or 3.\n";
        return 1;
    }

    try {
        Tensor<__float128> tensor = (ndim == 2)
            ? generate_2d(n, n_peaks, halo_avg, seed)
            : generate_3d(n, n_peaks, halo_avg, seed);

        std::cerr << "Generated " << ndim << "D FP128 tensor, shape=["
                  << n;
        if (ndim == 3) std::cerr << "x" << n;
        std::cerr << "x" << n << "], peaks=" << n_peaks << "\n";

        write_tensor(output_file, tensor);
        std::cerr << "Written to " << output_file << "\n";

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
