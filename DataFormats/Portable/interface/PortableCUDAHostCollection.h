#ifndef DataFormats_Portable_interface_PortableCUDAHostCollection_h
#define DataFormats_Portable_interface_PortableCUDAHostCollection_h

#include <cstdlib>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

// generic SoA-based product in host memory
template <typename T>
class PortableCUDAHostCollection {
public:
  using Layout = T;
  using View = typename Layout::View;
  using ConstView = typename Layout::ConstView;
  using Buffer = cms::cuda::host::unique_ptr<std::byte[]>;

  PortableCUDAHostCollection() = default;

  PortableCUDAHostCollection(int32_t elements)
      // allocate pageable host memory
      : buffer_{cms::cuda::make_host_unique<std::byte[]>(Layout::computeDataSize(elements))},
        layout_{buffer_.get(), elements},
        view_{layout_},
        constView{layout_} {
    // make_host_unique for pageable host memory uses a default alignment of 128 bytes
    assert(reinterpret_cast<uintptr_t>(buffer_.get()) % Layout::alignment == 0);
  }

  PortableCUDAHostCollection(int32_t elements, cudaStream_t stream)
      // allocate pinned host memory, accessible by the current device
      : buffer_{cms::cuda::make_host_unique<std::byte[]>(Layout::computeDataSize(elements), stream)},
        layout_{buffer_.get(), elements},
        view_{layout_},
        constView_{layout_} {
    // CUDA pinned host memory uses a default alignment of at least 128 bytes
    assert(reinterpret_cast<uintptr_t>(buffer_.get()) % Layout::alignment == 0);
  }

  // non-copyable
  PortableCUDAHostCollection(PortableCUDAHostCollection const &) = delete;
  PortableCUDAHostCollection &operator=(PortableCUDAHostCollection const &) = delete;

  // movable
  PortableCUDAHostCollection(PortableCUDAHostCollection &&other) = default;
  PortableCUDAHostCollection &operator=(PortableCUDAHostCollection &&other) = default;

  // default destructor
  ~PortableCUDAHostCollection() = default;

  // access the Layout
  Layout &layout() { return layout_; }
  Layout const &layout() const { return layout_; }

  // access the View
  View &view() { return view_; }
  View const &view() const { return constView_; }

  View &operator*() { return view_; }
  View const &operator*() const { return constView_; }

  View *operator->() { return &view_; }
  View const *operator->() const { return &constView_; }

  Buffer &buffer() { return buffer_; }
  Buffer const &buffer() const { return buffer_; }

private:
  Buffer buffer_;
  Layout layout_;
  View view_;
  ConstView constView_;
};

// generic SoA-based product in host memory
template <typename LAYOUT0, typename LAYOUT1, typename VIEW, typename CONST_VIEW>
class PortableCUDAHostCollection {
public:
  using Layout0 = LAYOUT0;
  using Layout1 = LAYOUT1;
  using View = VIEW;
  using ConstView = CONST_VIEW;
  using Buffer = cms::cuda::host::unique_ptr<std::byte[]>;

  PortableCUDAHostCollection() = default;

  PortableCUDAHostCollection(int32_t elements)
      // allocate pageable host memory
      : buffer_{cms::cuda::make_host_unique<std::byte[]>(Layout0::computeDataSize(elements) +
                                                         Layout1::computeDataSize(elements))},
        layout0_{buffer_.get(), elements},
        layout1_{layout0_.metadate().nextByte(), elements},
        view_{layout0_, layout1_},
        constView_{layout0_, layout1_} {
    // make_host_unique for pageable host memory uses a default alignment of 128 bytes
    assert(reinterpret_cast<uintptr_t>(buffer_.get()) % Layout0::alignment == 0);
    assert(reinterpret_cast<uintptr_t>(layout0_.metadata().nextByte()) % Layout1::alignment == 0);
  }

  PortableCUDAHostCollection(int32_t elements, cudaStream_t stream)
      // allocate pinned host memory, accessible by the current device
      : buffer_{cms::cuda::make_host_unique<std::byte[]>(Layout::computeDataSize(elements), stream)},
        layout0_{buffer_.get(), elements},
        layout1_{layout0_.metadata().nextByte(), elements},
        view_{layout0_, layout1_},
        constView_{layout0_, layout1_} {
    // CUDA pinned host memory uses a default alignment of at least 128 bytes
    assert(reinterpret_cast<uintptr_t>(buffer_.get()) % Layout0::alignment == 0);
    assert(reinterpret_cast<uintptr_t>(layout0_.metadata().nextByte()) % Layout1::alignment == 0);
  }

  // non-copyable
  PortableCUDAHostCollection(PortableCUDAHostCollection const &) = delete;
  PortableCUDAHostCollection &operator=(PortableCUDAHostCollection const &) = delete;

  // movable
  PortableCUDAHostCollection(PortableCUDAHostCollection &&other) = default;
  PortableCUDAHostCollection &operator=(PortableCUDAHostCollection &&other) = default;

  // default destructor
  ~PortableCUDAHostCollection() = default;

  // access the Layouts
  Layout0 &layout0() { return layout0_; }
  Layout0 const &layout0() const { return layout0_; }

  Layout1 &layout1() { return layout1_; }
  Layout1 const &layout1() const { return layout1_; }

  // access the View
  View &view() { return view_; }
  ConstView const &view() const { return constView_; }

  View &operator*() { return view_; }
  ConstView const &operator*() const { return constView_; }

  View *operator->() { return &view_; }
  ConstView const *operator->() const { return &constView_; }

  Buffer &buffer() { return buffer_; }
  Buffer const &buffer() const { return buffer_; }

private:
  Buffer buffer_;
  Layout0 layout0_;
  Layout1 layout1_;
  View view_;
  ConstView constView_;
};

#endif  // DataFormats_Portable_interface_PortableCUDAHostCollection_h
