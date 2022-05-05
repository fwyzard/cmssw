#ifndef DataFormats_Portable_interface_PortableCollection_h
#define DataFormats_Portable_interface_PortableCollection_h

#include <optional>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/alpaka/host.h"

// generic SoA-based product
template <typename T, typename TDev>
class PortableCollection {
public:
  using Buffer = alpaka::Buf<TDev, std::byte, alpaka::DimInt<1u>, uint32_t>;

  PortableCollection() : buffer_{}, layout_{} {
#ifdef DEBUG_COLLECTION_CTOR_DTOR
    std::cout << __PRETTY_FUNCTION__ << " [this=" << this << "]" << std::endl;
#endif  // DEBUG_COLLECTION_CTOR_DTOR
  }

  PortableCollection(int32_t elements, TDev const &device)
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

  ~PortableCollection() {
#ifdef DEBUG_COLLECTION_CTOR_DTOR
    std::cout << __PRETTY_FUNCTION__ << " [this=" << this << "]" << std::endl;
#endif  // DEBUG_COLLECTION_CTOR_DTOR
  }

  // non-copyable
  PortableCollection(PortableCollection const &) = delete;
  PortableCollection &operator=(PortableCollection const &) = delete;

  // movable
#ifdef DEBUG_COLLECTION_CTOR_DTOR
  PortableCollection(PortableCollection &&other)
      : buffer_{std::move(other.buffer_)}, layout_{std::move(other.layout_)} {
    std::cout << __PRETTY_FUNCTION__ << " [this=" << this << "]" << std::endl;
  }
#else
  PortableCollection(PortableCollection &&other) = default;
#endif  // DEBUG_COLLECTION_CTOR_DTOR
  PortableCollection &operator=(PortableCollection &&other) = default;

  T &operator*() { return layout_; }

  T const &operator*() const { return layout_; }

  T *operator->() { return &layout_; }

  T const *operator->() const { return &layout_; }

  Buffer &buffer() { return *buffer_; }

  Buffer const &buffer() const { return *buffer_; }

#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  template <typename U>
  static void ROOTReadStreamer(PortableCollection *newObj, U onfile) {
    newObj->~PortableCollection();
    new (newObj) PortableCollection(onfile.layout_.size(), host);
    newObj->layout_.ROOTReadStreamer(onfile);
  }
#endif

private:
  std::optional<Buffer> buffer_;  //!
  T layout_;
};

#endif  // DataFormats_Portable_interface_PortableCollection_h
