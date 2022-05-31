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
  // XXX Addition of typedef for index types.
  // size_type for indices. Compatible with ROOT, but limited to 2G entries
  typedef typename T::size_type size_type;
  // byte_size_type for byte counts. Not creating an artificial limit (and not ROOT serialized).
  typedef typename T::byte_size_type byte_size_type;

  using Buffer = alpaka::Buf<TDev, std::byte, alpaka::DimInt<1u>, byte_size_type>;

  PortableDeviceCollection() : buffer_{}, layout_{} {
#ifdef DEBUG_COLLECTION_CTOR_DTOR
    std::cout << __PRETTY_FUNCTION__ << " [this=" << this << "]" << std::endl;
#endif  // DEBUG_COLLECTION_CTOR_DTOR
  }

  PortableDeviceCollection(size_type elements, TDev const &device)
      : buffer_{alpaka::allocBuf<std::byte, byte_size_type>(
            device, alpaka::Vec<alpaka::DimInt<1u>, byte_size_type>{T::computeDataSize(elements)})},
        layout_{buffer_->data(), elements},
        view_{layout_} {
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

  typename T::template TrivialView<> &operator*() { return view_; }

  typename T::template TrivialView<> const &operator*() const { return view_; }

  typename T::template TrivialView<> *operator->() { return &view_; }

  typename T::template TrivialView<> const *operator->() const { return &view_; }

  Buffer &buffer() { return *buffer_; }

  Buffer const &buffer() const { return *buffer_; }

private:
  std::optional<Buffer> buffer_;  //!
  T layout_;
  using View = typename T::template TrivialView<>;
  View view_;
};

#endif  // DataFormats_Portable_interfacePortableDeviceCollection_h
