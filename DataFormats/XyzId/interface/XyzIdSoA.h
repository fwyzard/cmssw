#pragma once

#include <cassert>
#include <cstdint>

#ifdef DEBUG_SOA_CTOR_DTOR
#include <iostream>
#endif

// SoA layout with x, y, z, id fields
class XyzIdSoA {
public:
  static constexpr size_t alignment = 128;  // align all fields to 128 bytes

  // constructor
  XyzIdSoA() : size_(0), buffer_(nullptr), x_(nullptr), y_(nullptr), z_(nullptr), id_(nullptr) {
#ifdef DEBUG_SOA_CTOR_DTOR
    std::cout << "XyzIdSoA default constructor" << std::endl;
#endif
  }

  XyzIdSoA(size_t size, void *buffer)
      : size_(size),
        buffer_(buffer),
        x_(reinterpret_cast<double *>(reinterpret_cast<intptr_t>(buffer_))),
        y_(reinterpret_cast<double *>(reinterpret_cast<intptr_t>(x_) + pad(size * sizeof(double)))),
        z_(reinterpret_cast<double *>(reinterpret_cast<intptr_t>(y_) + pad(size * sizeof(double)))),
        id_(reinterpret_cast<int32_t *>(reinterpret_cast<intptr_t>(z_) + pad(size * sizeof(double)))) {
#ifdef DEBUG_SOA_CTOR_DTOR
    std::cout << "XyzIdSoA constructor with " << size_ << " elements at 0x" << buffer_ << std::endl;
#endif
  }

#ifdef DEBUG_SOA_CTOR_DTOR
  ~XyzIdSoA() {
    if (buffer_) {
      std::cout << "XyzIdSoA destructor with " << size_ << " elements at 0x" << buffer_ << std::endl;
    } else {
      std::cout << "XyzIdSoA destructor wihout data" << std::endl;
    }
  }
#else
  ~XyzIdSoA() = default;
#endif

  // non-copyable
  XyzIdSoA(XyzIdSoA const &) = delete;
  XyzIdSoA &operator=(XyzIdSoA const &) = delete;

  // movable
#ifdef DEBUG_SOA_CTOR_DTOR
  XyzIdSoA(XyzIdSoA &&other)
      : size_(other.size_), buffer_(other.buffer_), x_(other.x_), y_(other.y_), z_(other.z_), id_(other.id_) {
    std::cout << "XyzIdSoA move constructor with " << size_ << " elements at 0x" << buffer_ << std::endl;
    other.buffer_ = nullptr;
  }

  XyzIdSoA &operator=(XyzIdSoA &&other) {
    size_ = other.size_;
    buffer_ = other.buffer_;
    x_ = other.x_;
    y_ = other.y_;
    z_ = other.z_;
    id_ = other.id_;
    std::cout << "XyzIdSoA move assignment with " << size_ << " elements at 0x" << buffer_ << std::endl;
    other.buffer_ = nullptr;
    return *this;
  }
#else
  XyzIdSoA(XyzIdSoA &&other) = default;
  XyzIdSoA &operator=(XyzIdSoA &&other) = default;
#endif

  // global accessors
  size_t size() const { return size_; }

  size_t extent() const { return compute_size(size_); }

  void *data() { return buffer_; }
  void const *data() const { return buffer_; }

  // element-wise accessors are not implemented for simplicity

  // field-wise accessors
  double const &x(size_t i) const {
    assert(i < size_);
    return x_[i];
  }

  double &x(size_t i) {
    assert(i < size_);
    return x_[i];
  }

  double const &y(size_t i) const {
    assert(i < size_);
    return y_[i];
  }

  double &y(size_t i) {
    assert(i < size_);
    return y_[i];
  }

  double const &z(size_t i) const {
    assert(i < size_);
    return z_[i];
  }

  double &z(size_t i) {
    assert(i < size_);
    return z_[i];
  }

  int32_t const &id(size_t i) const {
    assert(i < size_);
    return id_[i];
  }

  int32_t &id(size_t i) {
    assert(i < size_);
    return id_[i];
  }

  // utilities
  static constexpr size_t pad(size_t size) { return ((size + alignment - 1) / alignment * alignment); }

  // takes the size in elements, returns the size in bytes
  static constexpr size_t compute_size(size_t elements) {
    return pad(elements * sizeof(double)) +  // x
           pad(elements * sizeof(double)) +  // y
           pad(elements * sizeof(double)) +  // z
           elements * sizeof(int32_t);       // id - no need to pad the last field
  }

private:
  // non-owned memory
  size_t size_;
  void *buffer_;

  // layout
  double *x_;
  double *y_;
  double *z_;
  int32_t *id_;
};
