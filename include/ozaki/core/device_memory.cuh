#ifndef CORE_DEVICE_MEMORY_CUH
#define CORE_DEVICE_MEMORY_CUH

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "../gpu/gpu_checks.cuh"

#ifdef __CUDACC__

namespace core {

namespace detail {

template <typename T> constexpr void validate_device_memory_type() {
    static_assert(std::is_object_v<T>, "Device memory can only store object types");
    static_assert(!std::is_const_v<T>, "Device memory cannot own const element types");
    static_assert(!std::is_volatile_v<T>, "Device memory cannot own volatile element types");
    static_assert(std::is_trivially_copyable_v<T>,
                  "Device memory uses cudaMemcpy and requires trivially copyable types");
}

template <typename T> constexpr std::size_t checked_size_bytes(std::size_t count) {
    validate_device_memory_type<T>();
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::length_error("Device allocation size overflows size_t");
    }
    return count * sizeof(T);
}

inline void validate_pointer(const void *ptr, std::size_t count, const char *name) {
    if (count != 0 && ptr == nullptr) {
        throw std::invalid_argument(name);
    }
}

inline void validate_range(std::size_t owner_size, std::size_t offset, std::size_t count) {
    if (offset > owner_size || count > owner_size - offset) {
        throw std::out_of_range("Device memory copy range is out of bounds");
    }
}

} // namespace detail

template <typename T> class DeviceArray {
  public:
    using value_type = T;
    using size_type = std::size_t;
    using pointer = T *;
    using const_pointer = const T *;

    DeviceArray() noexcept = default;

    explicit DeviceArray(size_type size) {
        detail::validate_device_memory_type<T>();
        allocate(size);
    }

    explicit DeviceArray(std::span<const T> host_values) : DeviceArray(host_values.size()) {
        copy_from_host(host_values);
    }

    DeviceArray(const T *host_values, size_type count) : DeviceArray(count) { copy_from_host(host_values, count); }

    DeviceArray(const DeviceArray &) = delete;
    DeviceArray &operator=(const DeviceArray &) = delete;

    DeviceArray(DeviceArray &&other) noexcept
        : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

    DeviceArray &operator=(DeviceArray &&other) noexcept {
        if (this != &other) {
            free_noexcept();
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    ~DeviceArray() noexcept { free_noexcept(); }

    void reset() noexcept {
        free_noexcept();
        size_ = 0;
    }

    void clear() noexcept { reset(); }

    void reset(size_type new_size) {
        if (new_size == size_) {
            return;
        }

        DeviceArray replacement(new_size);
        swap(replacement);
    }

    void swap(DeviceArray &other) noexcept {
        using std::swap;
        swap(data_, other.data_);
        swap(size_, other.size_);
    }

    [[nodiscard]] pointer get() noexcept { return data_; }
    [[nodiscard]] const_pointer get() const noexcept { return data_; }
    [[nodiscard]] void *as_arg() noexcept { return static_cast<void *>(&data_); }
    [[nodiscard]] void *as_arg() const noexcept { return static_cast<void *>(const_cast<pointer *>(&data_)); }
    [[nodiscard]] pointer data() noexcept { return data_; }
    [[nodiscard]] const_pointer data() const noexcept { return data_; }
    [[nodiscard]] size_type size() const noexcept { return size_; }
    [[nodiscard]] size_type size_bytes() const noexcept { return size_ * sizeof(T); }
    [[nodiscard]] size_type bytes() const noexcept { return size_bytes(); }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] explicit operator bool() const noexcept { return data_ != nullptr; }

    void zero() {
        if (size_ == 0) {
            return;
        }
        CUDA_CHECK(cudaMemset(data_, 0, bytes()));
    }

    void copy_from_host(std::span<const T> source) {
        if (source.size() != size_) {
            throw std::invalid_argument("Host source span size does not match DeviceArray size");
        }
        copy_from_host(source.data(), source.size());
    }

    void copy_from_host(const T *source, size_type count, size_type device_offset = 0) {
        detail::validate_range(size_, device_offset, count);
        detail::validate_pointer(source, count, "Host source pointer is null");
        if (count == 0) {
            return;
        }

        CUDA_CHECK(
            cudaMemcpy(data_ + device_offset, source, detail::checked_size_bytes<T>(count), cudaMemcpyHostToDevice));
    }

    void copy_to_host(std::span<T> destination) const {
        if (destination.size() != size_) {
            throw std::invalid_argument("Host destination span size does not match DeviceArray size");
        }
        copy_to_host(destination.data(), destination.size());
    }

    void copy_to_host(T *destination, size_type count, size_type device_offset = 0) const {
        detail::validate_range(size_, device_offset, count);
        detail::validate_pointer(destination, count, "Host destination pointer is null");
        if (count == 0) {
            return;
        }

        CUDA_CHECK(cudaMemcpy(destination, data_ + device_offset, detail::checked_size_bytes<T>(count),
                              cudaMemcpyDeviceToHost));
    }

    [[nodiscard]] std::vector<T> copy_to_host() const {
        std::vector<T> host_values(size_);
        copy_to_host(std::span<T>(host_values.data(), host_values.size()));
        return host_values;
    }

    void copy_from_device(const DeviceArray &source) {
        if (source.size() != size_) {
            throw std::invalid_argument("Device source array size does not match DeviceArray size");
        }
        copy_from_device(source.get(), source.size());
    }

    void copy_from_device(const T *source, size_type count, size_type device_offset = 0) {
        detail::validate_range(size_, device_offset, count);
        detail::validate_pointer(source, count, "Device source pointer is null");
        if (count == 0) {
            return;
        }

        CUDA_CHECK(
            cudaMemcpy(data_ + device_offset, source, detail::checked_size_bytes<T>(count), cudaMemcpyDeviceToDevice));
    }

    void copy_to_device(DeviceArray &destination) const { destination.copy_from_device(*this); }

    void copy_to_device(T *destination, size_type count, size_type source_offset = 0) const {
        detail::validate_range(size_, source_offset, count);
        detail::validate_pointer(destination, count, "Device destination pointer is null");
        if (count == 0) {
            return;
        }

        CUDA_CHECK(cudaMemcpy(destination, data_ + source_offset, detail::checked_size_bytes<T>(count),
                              cudaMemcpyDeviceToDevice));
    }

  private:
    void allocate(size_type count) {
        const size_type bytes = detail::checked_size_bytes<T>(count);
        if (bytes == 0) {
            return;
        }

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data_), bytes));
        size_ = count;
    }

    void free_noexcept() noexcept {
        if (data_ != nullptr) {
            (void)cudaFree(data_);
            data_ = nullptr;
        }
        size_ = 0;
    }

    pointer data_ = nullptr;
    size_type size_ = 0;
};

template <typename T> class DeviceObject {
  public:
    using value_type = T;
    using size_type = std::size_t;
    using pointer = T *;
    using const_pointer = const T *;

    DeviceObject() { allocate(); }

    explicit DeviceObject(const T &host_value) : DeviceObject() { copy_from_host(host_value); }

    DeviceObject(const DeviceObject &) = delete;
    DeviceObject &operator=(const DeviceObject &) = delete;

    DeviceObject(DeviceObject &&other) noexcept : data_(std::exchange(other.data_, nullptr)) {}

    DeviceObject &operator=(DeviceObject &&other) noexcept {
        if (this != &other) {
            free_noexcept();
            data_ = std::exchange(other.data_, nullptr);
        }
        return *this;
    }

    ~DeviceObject() noexcept { free_noexcept(); }

    void reset() {
        DeviceObject replacement;
        swap(replacement);
    }

    void reset(const T &host_value) {
        DeviceObject replacement(host_value);
        swap(replacement);
    }

    void clear() noexcept { free_noexcept(); }

    void swap(DeviceObject &other) noexcept {
        using std::swap;
        swap(data_, other.data_);
    }

    [[nodiscard]] pointer get() noexcept { return data_; }
    [[nodiscard]] const_pointer get() const noexcept { return data_; }
    [[nodiscard]] void *as_arg() noexcept { return static_cast<void *>(&data_); }
    [[nodiscard]] void *as_arg() const noexcept { return static_cast<void *>(const_cast<pointer *>(&data_)); }
    [[nodiscard]] pointer data() noexcept { return data_; }
    [[nodiscard]] const_pointer data() const noexcept { return data_; }
    [[nodiscard]] size_type size() const noexcept { return data_ == nullptr ? 0 : 1; }
    [[nodiscard]] size_type size_bytes() const noexcept { return size() * sizeof(T); }
    [[nodiscard]] size_type bytes() const noexcept { return size_bytes(); }
    [[nodiscard]] bool empty() const noexcept { return data_ == nullptr; }
    [[nodiscard]] explicit operator bool() const noexcept { return data_ != nullptr; }

    void copy_from_host(const T &source) {
        ensure_allocated();
        CUDA_CHECK(cudaMemcpy(data_, &source, sizeof(T), cudaMemcpyHostToDevice));
    }

    void copy_to_host(T &destination) const {
        ensure_allocated();
        CUDA_CHECK(cudaMemcpy(&destination, data_, sizeof(T), cudaMemcpyDeviceToHost));
    }

    [[nodiscard]] T copy_to_host() const {
        T destination{};
        copy_to_host(destination);
        return destination;
    }

    void copy_from_device(const DeviceObject &source) {
        ensure_allocated();
        source.ensure_allocated();
        CUDA_CHECK(cudaMemcpy(data_, source.get(), sizeof(T), cudaMemcpyDeviceToDevice));
    }

    void copy_to_device(DeviceObject &destination) const { destination.copy_from_device(*this); }

  private:
    void allocate() {
        detail::validate_device_memory_type<T>();
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data_), sizeof(T)));
    }

    void ensure_allocated() const {
        if (data_ == nullptr) {
            throw std::logic_error("DeviceObject does not own device memory");
        }
    }

    void free_noexcept() noexcept {
        if (data_ != nullptr) {
            (void)cudaFree(data_);
            data_ = nullptr;
        }
    }

    pointer data_ = nullptr;
};

template <typename T> [[nodiscard]] DeviceArray<T> make_device_array(std::size_t size) { return DeviceArray<T>(size); }

template <typename T> [[nodiscard]] DeviceObject<T> make_device_object() { return DeviceObject<T>(); }

template <typename T> [[nodiscard]] DeviceObject<T> make_device_object(const T &value) {
    return DeviceObject<T>(value);
}

} // namespace core

#endif // __CUDACC__

#endif // CORE_DEVICE_MEMORY_CUH
