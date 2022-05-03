#ifndef DataFormats_Portable_interface_AlpakaHostCollection_h
#define DataFormats_Portable_interface_AlpakaHostCollection_h

#include <optional>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/AlpakaCollection.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/config.h"

// generic SoA-based product in pinned host memory
template <typename T>
class AlpakaCollection<T, alpaka_common::DevHost> {
public:
  using Buffer = alpaka::Buf<alpaka_common::DevHost, std::byte, alpaka::DimInt<1u>, uint32_t>;

  AlpakaCollection() : buffer_{}, layout_{} {}

  AlpakaCollection(int32_t elements, alpaka_common::DevHost const &host)
      : buffer_{alpaka::allocBuf<std::byte, uint32_t>(
            host, alpaka::Vec<alpaka::DimInt<1u>, uint32_t>{T::compute_size(elements)})},
        layout_{elements, buffer_->data()} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % T::alignment == 0);
  }

  template <typename TDev>
  AlpakaCollection(int32_t elements, alpaka_common::DevHost const &host, TDev const &device)
      : buffer_{alpaka::allocMappedBuf<std::byte, uint32_t>(
            host, device, alpaka::Vec<alpaka::DimInt<1u>, uint32_t>{T::compute_size(elements)})},
        layout_{elements, buffer_->data()} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % T::alignment == 0);
  }

  ~AlpakaCollection() = default;

  // non-copyable
  AlpakaCollection(AlpakaCollection const &) = delete;
  AlpakaCollection &operator=(AlpakaCollection const &) = delete;

  // movable
  AlpakaCollection(AlpakaCollection &&other) = default;
  AlpakaCollection &operator=(AlpakaCollection &&other) = default;

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

template <typename T>
using AlpakaHostCollection = AlpakaCollection<T, alpaka_common::DevHost>;

#endif  // DataFormats_Portable_interface_AlpakaHostCollection_h
