// Owning wrappers for device allocations of trivially copyable types.
#ifndef OZAKI_CORE_DEVICE_MEMORY_CUH
#define OZAKI_CORE_DEVICE_MEMORY_CUH

#include <ozaki/gpu/gpu_checks.cuh>

#include <cstddef>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace core {

template <typename T> class DeviceArray {
  static_assert(std::is_trivially_copyable_v<T>);

public:
  DeviceArray() = default;
  explicit DeviceArray(size_t size) { reset(size); }
  explicit DeviceArray(std::span<const T> values) : DeviceArray(values.size()) { copy_from_host(values); }
  DeviceArray(DeviceArray &&other) noexcept { swap(other); }
  DeviceArray &operator=(DeviceArray &&other) noexcept {
    swap(other);
    return *this;
  }
  DeviceArray(const DeviceArray &) = delete;
  DeviceArray &operator=(const DeviceArray &) = delete;
  ~DeviceArray() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
  }

  // Replaces the allocation by an uninitialized one of `size` elements.
  void reset(size_t size) {
    DeviceArray fresh;
    if (size != 0) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&fresh.data_), size * sizeof(T)));
      fresh.size_ = size;
    }
    swap(fresh);
  }
  void swap(DeviceArray &other) noexcept {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
  }

  [[nodiscard]] T *get() const noexcept { return data_; }
  [[nodiscard]] size_t size() const noexcept { return size_; }
  [[nodiscard]] size_t size_bytes() const noexcept { return size_ * sizeof(T); }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

  void zero() { CUDA_CHECK(cudaMemset(data_, 0, size_bytes())); }
  void copy_from_host(std::span<const T> values) {
    if (values.size() != size_) {
      throw std::length_error("DeviceArray::copy_from_host size mismatch.");
    }
    CUDA_CHECK(cudaMemcpy(data_, values.data(), size_bytes(), cudaMemcpyHostToDevice));
  }

private:
  T *data_ = nullptr;
  size_t size_ = 0;
};

template <typename T> class DeviceObject {
public:
  DeviceObject() : storage_(1) {}
  explicit DeviceObject(const T &value) : DeviceObject() { copy_from_host(value); }

  [[nodiscard]] T *get() const noexcept { return storage_.get(); }
  void copy_from_host(const T &value) { storage_.copy_from_host(std::span<const T>(&value, 1)); }
  [[nodiscard]] T copy_to_host() const {
    T value;
    CUDA_CHECK(cudaMemcpy(&value, storage_.get(), sizeof(T), cudaMemcpyDeviceToHost));
    return value;
  }

private:
  DeviceArray<T> storage_;
};

} // namespace core

#endif // OZAKI_CORE_DEVICE_MEMORY_CUH
