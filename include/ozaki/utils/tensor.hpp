#ifndef HEADER_TENSOR_HPP
#define HEADER_TENSOR_HPP

#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

template <typename T> struct Tensor {
  std::vector<size_t> shape;
  std::vector<T> data;

  size_t size() const { return data.size(); }
};

template <typename T> size_t get_type_id() {
  if constexpr (std::is_same_v<T, float>) {
    return 0;
  } else if constexpr (std::is_same_v<T, double>) {
    return 1;
  } else if constexpr (std::is_same_v<T, long double>) {
    return 2;
  } else {
    static_assert(std::is_same_v<T, __float128>, "Unsupported tensor element type.");
    return 3;
  }
}

template <typename T> Tensor<T> read_tensor(const std::string &filepath) {
  std::ifstream f(filepath, std::ios::in | std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("Could not open file at " + filepath + ".");
  }

  size_t n_dims;
  f.read(reinterpret_cast<char *>(&n_dims), sizeof(size_t));

  size_t type_id;
  f.read(reinterpret_cast<char *>(&type_id), sizeof(size_t));

  size_t expected_type_id = get_type_id<T>();
  if (type_id != expected_type_id) {
    throw std::runtime_error("Type mismatch: expected type " + std::to_string(expected_type_id) +
                             ", but file contains type " + std::to_string(type_id));
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

// Read a tensor file and widen/narrow its elements to type T regardless of the stored type.
template <typename T> Tensor<T> read_tensor_as(const std::string &filepath) {
  std::ifstream f(filepath, std::ios::in | std::ios::binary);
  if (!f.is_open())
    throw std::runtime_error("Could not open file at " + filepath + ".");

  size_t n_dims;
  f.read(reinterpret_cast<char *>(&n_dims), sizeof(size_t));

  size_t type_id;
  f.read(reinterpret_cast<char *>(&type_id), sizeof(size_t));

  std::vector<size_t> shape(n_dims);
  f.read(reinterpret_cast<char *>(shape.data()), n_dims * sizeof(size_t));

  size_t n_elems = 1;
  for (const auto &dim : shape)
    n_elems *= dim;

  // If the stored type matches T, read directly.
  if (type_id == get_type_id<T>()) {
    std::vector<T> data(n_elems);
    f.read(reinterpret_cast<char *>(data.data()), n_elems * sizeof(T));
    return {shape, data};
  }

  // Otherwise, read and convert element-by-element using scalar temporaries.
  // Avoids instantiating std::vector<StoredT>, which is necessary when StoredT
  // is __float128 in nvcc-compiled code (nvcc rejects std::initializer_list<__float128>
  // on sub-sm100 architectures).
  std::vector<T> data(n_elems);
  auto read_and_convert = [&]<typename StoredT>() {
    for (size_t i = 0; i < n_elems; ++i) {
      StoredT val;
      f.read(reinterpret_cast<char *>(&val), sizeof(StoredT));
      data[i] = static_cast<T>(val);
    }
  };

  switch (type_id) {
  case 0:
    read_and_convert.template operator()<float>();
    break;
  case 1:
    read_and_convert.template operator()<double>();
    break;
  case 2:
    read_and_convert.template operator()<long double>();
    break;
  case 3:
    read_and_convert.template operator()<__float128>();
    break;
  default:
    throw std::runtime_error("Unknown type_id " + std::to_string(type_id) + " in " + filepath);
  }
  return {shape, data};
}

// Write tensor data as OutT from a std::vector<InT>, converting element-by-element.
// Avoids instantiating std::vector<OutT>, which is necessary when OutT is __float128
// in nvcc-compiled code (nvcc rejects std::initializer_list<__float128> on sub-sm100).
template <typename OutT, typename InT>
void write_tensor_cast(const std::string &filepath, const std::vector<size_t> &shape, const std::vector<InT> &data) {
  std::ofstream f(filepath, std::ios::out | std::ios::binary);
  if (!f.is_open())
    throw std::runtime_error("Could not open file at " + filepath + ".");

  size_t n_dims = shape.size();
  f.write(reinterpret_cast<const char *>(&n_dims), sizeof(size_t));

  size_t type_id = get_type_id<OutT>();
  f.write(reinterpret_cast<const char *>(&type_id), sizeof(size_t));

  f.write(reinterpret_cast<const char *>(shape.data()), n_dims * sizeof(size_t));

  for (size_t i = 0; i < data.size(); ++i) {
    OutT val = static_cast<OutT>(data[i]);
    f.write(reinterpret_cast<const char *>(&val), sizeof(OutT));
  }
}

template <typename T> void write_tensor(const std::string &filepath, const Tensor<T> &tensor) {
  std::ofstream f(filepath, std::ios::out | std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("Could not open file at " + filepath + ".");
  }

  size_t n_dims = tensor.shape.size();
  f.write(reinterpret_cast<const char *>(&n_dims), sizeof(size_t));

  size_t type_id = get_type_id<T>();
  f.write(reinterpret_cast<const char *>(&type_id), sizeof(size_t));

  f.write(reinterpret_cast<const char *>(tensor.shape.data()), n_dims * sizeof(size_t));
  f.write(reinterpret_cast<const char *>(tensor.data.data()), tensor.size() * sizeof(T));
}

#endif // HEADER_TENSOR_HPP
