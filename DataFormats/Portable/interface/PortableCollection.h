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

  PortableCollection() : buffer_{}, layout_{} {}

  PortableCollection(int32_t elements, TDev const &device)
      : buffer_{alpaka::allocBuf<std::byte, uint32_t>(
            device, alpaka::Vec<alpaka::DimInt<1u>, uint32_t>{T::compute_size(elements)})},
        layout_{elements, buffer_->data()} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % T::alignment == 0);
    alpaka::pin(*buffer_);
  }

  ~PortableCollection() {}

  // non-copyable
  PortableCollection(PortableCollection const &) = delete;
  PortableCollection &operator=(PortableCollection const &) = delete;

  // movable
  PortableCollection(PortableCollection &&other) = default;
  PortableCollection &operator=(PortableCollection &&other) = default;

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

#endif  // DataFormats_Portable_interface_PortableCollection_h
