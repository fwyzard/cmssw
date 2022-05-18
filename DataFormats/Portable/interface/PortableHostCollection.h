#ifndef DataFormats_Portable_interface_PortableHostCollection_h
#define DataFormats_Portable_interface_PortableHostCollection_h

#include <optional>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"

// generic SoA-based product in host memory
template <typename T>
class PortableHostCollection {
public:
  using Buffer = alpaka::Buf<alpaka_common::DevHost, std::byte, alpaka::DimInt<1u>, uint32_t>;

  PortableHostCollection() : buffer_{}, layout_{} {
#ifdef DEBUG_COLLECTION_CTOR_DTOR
    std::cout << __PRETTY_FUNCTION__ << " [this=" << this << "]" << std::endl;
#endif  // DEBUG_COLLECTION_CTOR_DTOR
  }

  PortableHostCollection(int32_t elements, alpaka_common::DevHost const &host)
      // allocate pageable host memory
      : buffer_{alpaka::allocBuf<std::byte, uint32_t>(
            host, alpaka::Vec<alpaka::DimInt<1u>, uint32_t>{T::compute_size(elements)})},
        layout_{elements, buffer_->data()} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % T::alignment == 0);
#ifdef DEBUG_COLLECTION_CTOR_DTOR
    std::cout << __PRETTY_FUNCTION__ << " [this=" << this << "]" << std::endl;
#endif  // DEBUG_COLLECTION_CTOR_DTOR
  }

  template <typename TDev>
  PortableHostCollection(int32_t elements, alpaka_common::DevHost const &host, TDev const &device)
      // allocate pinned host memory, accessible by the given device
      : buffer_{alpaka::allocMappedBuf<std::byte, uint32_t>(
            host, device, alpaka::Vec<alpaka::DimInt<1u>, uint32_t>{T::compute_size(elements)})},
        layout_{elements, buffer_->data()} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % T::alignment == 0);
#ifdef DEBUG_COLLECTION_CTOR_DTOR
    std::cout << __PRETTY_FUNCTION__ << " [this=" << this << "]" << std::endl;
#endif  // DEBUG_COLLECTION_CTOR_DTOR
  }

  ~PortableHostCollection() {
#ifdef DEBUG_COLLECTION_CTOR_DTOR
    std::cout << __PRETTY_FUNCTION__ << " [this=" << this << "]" << std::endl;
#endif  // DEBUG_COLLECTION_CTOR_DTOR
  }

  // non-copyable
  PortableHostCollection(PortableHostCollection const &) = delete;
  PortableHostCollection &operator=(PortableHostCollection const &) = delete;

  // movable
#ifdef DEBUG_COLLECTION_CTOR_DTOR
  PortableHostCollection(PortableHostCollection &&other)
      : buffer_{std::move(other.buffer_)}, layout_{std::move(other.layout_)} {
    std::cout << __PRETTY_FUNCTION__ << " [this=" << this << "]" << std::endl;
  }
#else
  PortableHostCollection(PortableHostCollection &&other) = default;
#endif  // DEBUG_COLLECTION_CTOR_DTOR
  PortableHostCollection &operator=(PortableHostCollection &&other) = default;

  T &operator*() { return layout_; }

  T const &operator*() const { return layout_; }

  T *operator->() { return &layout_; }

  T const *operator->() const { return &layout_; }

  Buffer &buffer() { return *buffer_; }

  Buffer const &buffer() const { return *buffer_; }
  
  bool hasBuffer() const { return buffer_.operator bool(); }

  // part of the ROOT read streamer
  template <typename U>
  static void ROOTReadStreamer(PortableHostCollection *newObj, U onfile) {
    newObj->~PortableHostCollection();
    // use the global "host" object defined in HeterogeneousCore/AlpakaInterface/interface/host.h
    new (newObj) PortableHostCollection(onfile.layout_.size(), host);
    newObj->layout_.ROOTReadStreamer(onfile);
  }

private:
  std::optional<Buffer> buffer_;  //!
  T layout_;
};

#endif  // DataFormats_Portable_interface_PortableHostCollection_h
