#pragma once

#include <cassert>
#include <cstdint>

#ifdef DEBUG_SOA_CTOR_DTOR
#include <iostream>
#endif

#include "FWCore/Utilities/interface/typedefs.h"

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

  XyzIdSoA(int32_t size, void *buffer)
      : size_(size),
        buffer_(buffer),
        x_(reinterpret_cast<double *>(reinterpret_cast<intptr_t>(buffer_))),
        y_(reinterpret_cast<double *>(reinterpret_cast<intptr_t>(x_) + pad(size * sizeof(double)))),
        z_(reinterpret_cast<double *>(reinterpret_cast<intptr_t>(y_) + pad(size * sizeof(double)))),
        id_(reinterpret_cast<int32_t *>(reinterpret_cast<intptr_t>(z_) + pad(size * sizeof(double)))) {
    assert(size == 0 or (size > 0 and buffer != nullptr));
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
  int32_t size() const { return size_; }

  uint32_t extent() const { return compute_size(size_); }

  void *data() { return buffer_; }
  void const *data() const { return buffer_; }

  // element-wise accessors are not implemented for simplicity

  // field-wise accessors
  double const &x(int32_t i) const {
    assert(i >= 0);
    assert(i < size_);
    return x_[i];
  }

  double &x(int32_t i) {
    assert(i >= 0);
    assert(i < size_);
    return x_[i];
  }

  double const &y(int32_t i) const {
    assert(i >= 0);
    assert(i < size_);
    return y_[i];
  }

  double &y(int32_t i) {
    assert(i >= 0);
    assert(i < size_);
    return y_[i];
  }

  double const &z(int32_t i) const {
    assert(i >= 0);
    assert(i < size_);
    return z_[i];
  }

  double &z(int32_t i) {
    assert(i >= 0);
    assert(i < size_);
    return z_[i];
  }

  int32_t const &id(int32_t i) const {
    assert(i >= 0);
    assert(i < size_);
    return id_[i];
  }

  int32_t &id(int32_t i) {
    assert(i >= 0);
    assert(i < size_);
    return id_[i];
  }

  // pad a size (in bytes) to the next multiple of the alignment
  static constexpr uint32_t pad(size_t size) { return ((size + alignment - 1) / alignment * alignment); }

  // takes the size in elements, returns the size in bytes
  static constexpr uint32_t compute_size(int32_t elements) {
    assert(elements >= 0);
    return pad(elements * sizeof(double)) +  // x
           pad(elements * sizeof(double)) +  // y
           pad(elements * sizeof(double)) +  // z
           elements * sizeof(int32_t);       // id - no need to pad the last field
  }

private:
  // non-owned memory
  cms_int32_t size_;  // must be the same as ROOT's Int_t
  void *buffer_;      //!

  // layout
  double *x_;    //[size_]
  double *y_;    //[size_]
  double *z_;    //[size_]
  int32_t *id_;  //[size_]
};
