#ifndef DataFormats_Portable_interface_PortableCUDADeviceCollection_h
#define DataFormats_Portable_interface_PortableCUDADeviceCollection_h

#include <optional>
#include <type_traits>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

// generic SoA-based product in device memory
template <typename T>
class PortableCUDADeviceCollection {

public:
  using Layout = T;
  using View = typename Layout::View;
  using ConstView = typename Layout::ConstView;
  using Buffer = cms::cuda::device::unique_ptr<std::byte[]>;

  PortableCUDADeviceCollection() = default;

  PortableCUDADeviceCollection(int32_t elements, cudaStream_t stream)
      : buffer_{cms::cuda::make_device_unique<std::byte[]>(Layout::computeDataSize(elements), stream)},
        layout_{buffer_.get(), elements},
        view_{layout_},
        constView_{layout_}{
    // CUDA device memory uses a default alignment of at least 128 bytes
    assert(reinterpret_cast<uintptr_t>(buffer_.get()) % Layout::alignment == 0);
  }

  // non-copyable
  PortableCUDADeviceCollection(PortableCUDADeviceCollection const &) = delete;
  PortableCUDADeviceCollection &operator=(PortableCUDADeviceCollection const &) = delete;

  // movable
  PortableCUDADeviceCollection(PortableCUDADeviceCollection &&other) = default;
  PortableCUDADeviceCollection &operator=(PortableCUDADeviceCollection &&other) = default;

  // default destructor
  ~PortableCUDADeviceCollection() = default;

  // access the View
  View &view() { return view_; } 
  ConstView const &view() const { return constView_; } 

  View &operator*() { return view_; }
  ConstView const &operator*() const { return constView_; }

  View *operator->() { return &view_; }
  ConstView const *operator->() const { return &constView_; }

  Buffer::element_type *buffer() { return buffer_.get(); }
  Buffer::element_type const *buffer() const { return buffer_.get(); }

private:
  Buffer buffer_;
  Layout layout_;
  View view_;
  ConstView constView_;
};

// generic SoA-based product in host memory
template <typename LAYOUT0, typename LAYOUT1, typename VIEW, typename CONST_VIEW>
class PortableCUDADeviceCollection_2layouts {
public:
  using Layout0 = LAYOUT0;
  using Layout1 = LAYOUT1;
  using View = VIEW;
  using ConstView = CONST_VIEW;
  using Buffer = cms::cuda::host::unique_ptr<std::byte[]>;

  PortableCUDADeviceCollection_2layouts() = default;

  PortableCUDADeviceCollection_2layouts(int32_t elements)
      // allocate pageable host memory
      : buffer_{cms::cuda::make_host_unique<std::byte[]>(Layout0::computeDataSize(elements)+Layout1::computeDataSize(elements))},
        layout0_{buffer_.get(), elements},
        layout1_{layout0_.metadate().nextByte(), elements},
        view_{layout0_, layout1_},
        constView_ {layout0_, layout1_} {
    // make_host_unique for pageable host memory uses a default alignment of 128 bytes
    assert(reinterpret_cast<uintptr_t>(buffer_.get()) % Layout0::alignment == 0);
    assert(reinterpret_cast<uintptr_t>(layout0_.metadata().nextByte()) % Layout1::alignment == 0);
  }

  PortableCUDADeviceCollection_2layouts(int32_t elements, cudaStream_t stream)
      // allocate pinned host memory, accessible by the current device
      : buffer_{cms::cuda::make_host_unique<std::byte[]>(Layout0::computeDataSize(elements)+Layout1::computeDataSize(elements), stream)},
        layout0_{buffer_.get(), elements},
        layout1_{layout0_.metadata().nextByte(), elements},
        view_{layout0_, layout1_},
        constView_{layout0_, layout1_} {
    // CUDA pinned host memory uses a default alignment of at least 128 bytes
    assert(reinterpret_cast<uintptr_t>(buffer_.get()) % Layout0::alignment == 0);
    assert(reinterpret_cast<uintptr_t>(layout0_.metadata().nextByte()) % Layout1::alignment == 0);
  }

  // non-copyable
  PortableCUDADeviceCollection_2layouts(PortableCUDADeviceCollection_2layouts const &) = delete;
  PortableCUDADeviceCollection_2layouts &operator=(PortableCUDADeviceCollection_2layouts const &) = delete;

  // movable
  PortableCUDADeviceCollection_2layouts(PortableCUDADeviceCollection_2layouts &&other) = default;
  PortableCUDADeviceCollection_2layouts &operator=(PortableCUDADeviceCollection_2layouts &&other) = default;

  // default destructor
  ~PortableCUDADeviceCollection_2layouts() = default;

  // access the View
  View &view() { return view_; } 
  ConstView const &view() const { return constView_; } 

  View &operator*() { return view_; }
  ConstView const &operator*() const { return constView_; }

  View *operator->() { return &view_; }
  ConstView const *operator->() const { return &constView_; }

  Buffer &buffer() { return buffer_; }
  Buffer const &buffer() const { return buffer_; }  
  
  Layout0 &layout0() { return layout0_; }
  Layout0 const &layout0() const { return layout0_; }

  Layout0 &layout1() { return layout1_; }
  Layout0 const &layout1() const { return layout1_; }

  private:
  Buffer buffer_;
  Layout0 layout0_;
  Layout1 layout1_;
  View view_;
  ConstView constView_;
};


#endif  // DataFormats_Portable_interface_PortableCUDADeviceCollection_h

