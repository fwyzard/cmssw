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
  // XXX Addition of typedef for index types.
  // size_type for indices. Compatible with ROOT, but limited to 2G entries
  typedef typename T::size_type size_type;
  // byte_size_type for byte counts. Not creating an artificial limit (and not ROOT serialized).
  typedef typename T::byte_size_type byte_size_type;

  using Buffer = alpaka::Buf<alpaka_common::DevHost, std::byte, alpaka::DimInt<1u>, byte_size_type>;

  PortableHostCollection() : buffer_{}, layout_{}, view_{} {
#ifdef DEBUG_COLLECTION_CTOR_DTOR
    std::cout << __PRETTY_FUNCTION__ << " [this=" << this << "]" << std::endl;
#endif  // DEBUG_COLLECTION_CTOR_DTOR
  }

  PortableHostCollection(size_type elements, alpaka_common::DevHost const &host)
      : buffer_{alpaka::allocBuf<std::byte, byte_size_type>(
            host, alpaka::Vec<alpaka::DimInt<1u>, byte_size_type>{T::computeDataSize(elements)})},
        layout_{buffer_->data(), elements},
        view_{layout_} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % T::byteAlignment == 0);
#ifdef DEBUG_COLLECTION_CTOR_DTOR
    std::cout << __PRETTY_FUNCTION__ << " [this=" << this << "]" << std::endl;
#endif  // DEBUG_COLLECTION_CTOR_DTOR
  }

  template <typename TDev>
  PortableHostCollection(size_type elements, alpaka_common::DevHost const &host, TDev const &device)
      : buffer_{alpaka::allocMappedBuf<std::byte, byte_size_type>(
            host, device, alpaka::Vec<alpaka::DimInt<1u>, byte_size_type>{T::computeDataSize(elements)})},
        layout_{buffer_->data(), elements},
        view_(layout_) {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % T::byteAlignment == 0);
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

  typename T::template TrivialView<> &operator*() { return view_; }

  typename T::template TrivialView<> const &operator*() const { return view_; }

  typename T::template TrivialView<> *operator->() { return &view_; }

  typename T::template TrivialView<> const *operator->() const { return &view_; }

  Buffer &buffer() { return *buffer_; }

  Buffer const &buffer() const { return *buffer_; }

  bool hasBuffer() const { return buffer_.operator bool(); }

  // part of the ROOT read streamer
  template <typename U>
  static void ROOTReadStreamer(PortableHostCollection *newObj, U onfile) {
    newObj->~PortableHostCollection();
    // use the global "host" object defined in HeterogeneousCore/AlpakaInterface/interface/host.h
    new (newObj) PortableHostCollection(onfile.layout_.soaMetadata().size(), host);
    newObj->layout_.ROOTReadStreamer(onfile);
  }

private:
  std::optional<Buffer> buffer_;  //!
  T layout_;
  using View = typename T::template TrivialView<>;
  View view_;
};

#endif  // DataFormats_Portable_interface_PortableHostCollection_h
