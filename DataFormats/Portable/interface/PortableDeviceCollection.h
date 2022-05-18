#ifndef DataFormats_Portable_interfacePortableDeviceCollection_h
#define DataFormats_Portable_interfacePortableDeviceCollection_h

#include <optional>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// generic SoA-based product in device memory
template <typename T, typename TDev>
class PortableDeviceCollection {
  static_assert(not std::is_same_v<TDev, alpaka_common::DevHost>,
                "Use PortableHostCollection<T> instead of PortableDeviceCollection<T, DevHost>");

public:
  using Buffer = alpaka::Buf<TDev, std::byte, alpaka::DimInt<1u>, uint32_t>;

  PortableDeviceCollection() : buffer_{}, layout_{} {
#ifdef DEBUG_COLLECTION_CTOR_DTOR
    std::cout << __PRETTY_FUNCTION__ << " [this=" << this << "]" << std::endl;
#endif  // DEBUG_COLLECTION_CTOR_DTOR
  }

  PortableDeviceCollection(int32_t elements, TDev const &device)
      : buffer_{alpaka::allocBuf<std::byte, uint32_t>(
            device, alpaka::Vec<alpaka::DimInt<1u>, uint32_t>{T::compute_size(elements)})},
        layout_{elements, buffer_->data()} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % T::alignment == 0);
    alpaka::pin(*buffer_);
#ifdef DEBUG_COLLECTION_CTOR_DTOR
    std::cout << __PRETTY_FUNCTION__ << " [this=" << this << "]" << std::endl;
#endif  // DEBUG_COLLECTION_CTOR_DTOR
  }

  ~PortableDeviceCollection() {
#ifdef DEBUG_COLLECTION_CTOR_DTOR
    std::cout << __PRETTY_FUNCTION__ << " [this=" << this << "]" << std::endl;
#endif  // DEBUG_COLLECTION_CTOR_DTOR
  }

  // non-copyable
  PortableDeviceCollection(PortableDeviceCollection const &) = delete;
  PortableDeviceCollection &operator=(PortableDeviceCollection const &) = delete;

  // movable
#ifdef DEBUG_COLLECTION_CTOR_DTOR
  PortableDeviceCollection(PortableDeviceCollection &&other)
      : buffer_{std::move(other.buffer_)}, layout_{std::move(other.layout_)} {
    std::cout << __PRETTY_FUNCTION__ << " [this=" << this << "]" << std::endl;
  }
#else
  PortableDeviceCollection(PortableDeviceCollection &&other) = default;
#endif  // DEBUG_COLLECTION_CTOR_DTOR
  PortableDeviceCollection &operator=(PortableDeviceCollection &&other) = default;

  T &operator*() { return layout_; }

  T const &operator*() const { return layout_; }

  T *operator->() { return &layout_; }

  T const *operator->() const { return &layout_; }

  Buffer &buffer() { return *buffer_; }

  Buffer const &buffer() const { return *buffer_; }

private:
  std::optional<Buffer> buffer_;  //!
  T layout_;
};

#endif  // DataFormats_Portable_interfacePortableDeviceCollection_h
