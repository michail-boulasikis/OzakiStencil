#ifndef HEADER_TENSOR_HPP
#define HEADER_TENSOR_HPP

#include <vector>
#include <fstream>
#include <format>
#include <type_traits>

template <typename T> struct Tensor {
  std::vector<size_t> shape;
  std::vector<T> data;

  size_t size() const { return data.size(); }
  size_t width() const { return shape.empty() ? 0 : shape[0]; }
  size_t height() const { return shape.size() > 1 ? shape[1] : 1; }
};

template <typename T> size_t get_type_id() {
  if constexpr (std::is_same_v<T, float>) {
    return 0;
  } else if constexpr (std::is_same_v<T, double>) {
    return 1;
  } else if constexpr (std::is_same_v<T, long double>) {
    return 2;
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 100)
  } else if constexpr (std::is_same_v<T, __float128>) {
    return 3;
  }
#else
  } else {
    return 3;
  }
#endif

}

template <typename T> Tensor<T> read_tensor(const std::string &filepath) {
  std::ifstream f(filepath, std::ios::in | std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error(std::format("Could not open file at {}.", filepath));
  }

  size_t n_dims;
  f.read(reinterpret_cast<char *>(&n_dims), sizeof(size_t));

  size_t type_id;
  f.read(reinterpret_cast<char *>(&type_id), sizeof(size_t));

  size_t expected_type_id = get_type_id<T>();
  if (type_id != expected_type_id) {
    throw std::runtime_error(std::format("Type mismatch: expected type {}, but file contains type {}", expected_type_id, type_id));
  }

  std::vector<size_t> shape(n_dims);
  f.read(reinterpret_cast<char *>(shape.data()), n_dims * sizeof(size_t));

  size_t n_elems = 1;
  for (const auto &dim : shape)
    n_elems *= dim;

  std::vector<T> data(n_elems);
  f.read(reinterpret_cast<char *>(data.data()), n_elems * sizeof(T));

  return {shape, data};
}

template <typename T> void write_tensor(const std::string &filepath, const Tensor<T> &tensor) {
  std::ofstream f(filepath, std::ios::out | std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error(std::format("Could not open file at {}.", filepath));
  }

  size_t n_dims = tensor.shape.size();
  f.write(reinterpret_cast<const char *>(&n_dims), sizeof(size_t));

  size_t type_id = get_type_id<T>();
  f.write(reinterpret_cast<const char *>(&type_id), sizeof(size_t));

  f.write(reinterpret_cast<const char *>(tensor.shape.data()), n_dims * sizeof(size_t));
  f.write(reinterpret_cast<const char *>(tensor.data.data()), tensor.size() * sizeof(T));
}



#endif // HEADER_TENSOR_HPP

