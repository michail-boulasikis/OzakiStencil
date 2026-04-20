#include <ozaki/core/oztraits.hpp>
#include <ozaki/utils/tensor.hpp>

#include <boost/program_options.hpp>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace po = boost::program_options;

// Generate an fp128 value by sampling a double and adding sub-double noise.
static __float128 rand_fp128(std::mt19937_64 &rng,
                              std::normal_distribution<double> &gauss,
                              std::uniform_real_distribution<double> &unit) {
    double hi = gauss(rng);
    __float128 q = (__float128)hi;
    q += (__float128)hi * (__float128)(unit(rng) * 0x1p-53);
    q += (__float128)hi * (__float128)(unit(rng) * 0x1p-106);
    return q;
}

static Tensor<__float128> generate(std::vector<size_t> shape, bool halo_avg, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double>   gauss(0.0, 1.0);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);

    size_t total = 1;
    for (size_t s : shape) total *= s;

    std::vector<__float128> output(total);
    for (auto &v : output)
        v = rand_fp128(rng, gauss, unit);

    if (halo_avg) {
        __float128 sum = (__float128)0.0;
        for (auto v : output) sum += v;
        __float128 avg = sum / (__float128)total;

        if (shape.size() == 1) {
            size_t n = shape[0];
            output[0]     = avg;
            output[n - 1] = avg;
        } else if (shape.size() == 2) {
            size_t n = shape[0];
            for (size_t i = 0; i < n; ++i) {
                output[0 * n + i]       = avg;
                output[(n - 1) * n + i] = avg;
                output[i * n + 0]       = avg;
                output[i * n + (n - 1)] = avg;
            }
        } else if (shape.size() == 3) {
            size_t n = shape[0];
            for (size_t j = 0; j < n; ++j)
                for (size_t i = 0; i < n; ++i) {
                    output[0 * n * n + j * n + i]       = avg;
                    output[(n-1) * n * n + j * n + i]   = avg;
                }
            for (size_t k = 0; k < n; ++k)
                for (size_t i = 0; i < n; ++i) {
                    output[k * n * n + 0 * n + i]       = avg;
                    output[k * n * n + (n-1) * n + i]   = avg;
                }
            for (size_t k = 0; k < n; ++k)
                for (size_t j = 0; j < n; ++j) {
                    output[k * n * n + j * n + 0]       = avg;
                    output[k * n * n + j * n + (n-1)]   = avg;
                }
        }
    }

    return {shape, std::move(output)};
}

int main(int argc, char *argv[]) {
    std::string output_file;
    size_t      n        = 64;
    int         ndim     = 2;
    uint64_t    seed     = 42;
    bool        halo_avg = false;

    po::options_description desc("gen-tensor: generate FP128 Gaussian random tensor files");
    // clang-format off
    desc.add_options()
        ("help,h",   "Produce help message")
        ("output,o", po::value<std::string>(&output_file)->required(),
                     "Path to output binary tensor file")
        ("N",        po::value<size_t>(&n)->default_value(64),
                     "Grid size per dimension")
        ("dims,d",   po::value<int>(&ndim)->default_value(2),
                     "Dimensionality: 1, 2, or 3")
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

    if (ndim < 1 || ndim > 3) {
        std::cerr << "Error: --dims must be 1, 2, or 3.\n";
        return 1;
    }

    try {
        std::vector<size_t> shape(ndim, n);
        Tensor<__float128>  tensor = generate(shape, halo_avg, seed);

        std::cerr << "Generated " << ndim << "D FP128 tensor, shape=[" << n;
        for (int d = 1; d < ndim; ++d) std::cerr << "x" << n;
        std::cerr << "]\n";

        write_tensor(output_file, tensor);
        std::cerr << "Written to " << output_file << "\n";

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
