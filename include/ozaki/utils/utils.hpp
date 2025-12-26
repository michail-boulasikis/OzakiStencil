#ifndef HEADER_UTILS_HPP
#define HEADER_UTILS_HPP

#include "../core/types.hpp"
#include <array>
#include <filesystem>
#include <fstream>

void print_bin_f64(const float64_t a);

template <typename T, size_t ND>
void write_tensor(const std::filesystem::path &fpath, const T* data, const std::array<size_t, ND> &dims) {
  std::ofstream fs(fpath, std::ios::out | std::ios::binary);
  size_t nd_value = ND;
  fs.write(reinterpret_cast<const char*>(&nd_value), sizeof(size_t));
  fs.write(reinterpret_cast<const char*>(dims.data()), sizeof(dims));
  size_t n_elems = 1;
  for (const auto &d : dims) n_elems *= d;
  fs.write(reinterpret_cast<const char*>(data), n_elems * sizeof(T));
  fs.close();
}


#endif // HEADER_UTILS_HPP
