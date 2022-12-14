#ifndef DataFormats_Portable_interface_PortableDeviceCollection_h
#define DataFormats_Portable_interface_PortableDeviceCollection_h

#include <cassert>
#include <optional>
#include <type_traits>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "DataFormats/Portable/interface/PortableCollectionCommon.h"

// generic SoA-based product in device memory
template <typename TDev,
          typename T0,
          typename T1 = void,
          typename T2 = void,
          typename T3 = void,
          typename T4 = void,
          typename = std::enable_if_t<cms::alpakatools::is_device_v<TDev>>>
class PortableDeviceCollection {
  static_assert(not std::is_same_v<TDev, alpaka_common::DevHost>,
                "Use PortableHostCollection<T> instead of PortableDeviceCollection<T, DevHost>");
  // Make sure void is not interleaved with other types.
  static_assert(not std::is_same<T3, void>::value or std::is_same<T4, void>::value);
  static_assert(not std::is_same<T2, void>::value or std::is_same<T3, void>::value);
  static_assert(not std::is_same<T1, void>::value or std::is_same<T2, void>::value);

  template <typename T>
  static constexpr std::size_t count_t_ = portablecollection::typeCount<T, T0, T1, T2, T3, T4>;

  template <typename T>
  static constexpr std::size_t index_t_ = portablecollection::typeIndex<T, T0, T1, T2, T3, T4>;

  static constexpr std::size_t members_ = portablecollection::membersCount<T0, T1, T2, T3, T4>;

public:
  using Buffer = cms::alpakatools::device_buffer<TDev, std::byte[]>;
  using ConstBuffer = cms::alpakatools::const_device_buffer<TDev, std::byte[]>;
  using Implementation = portablecollection::CollectionImpl<0, T0, T1, T2, T3, T4>;

  using SizesArray = std::array<int32_t, members_>;

  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  using Layout = portablecollection::TypeResolver<Idx, T0, T1, T2, T3, T4>;
  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  using View = typename Layout<Idx>::View;
  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  using ConstView = typename Layout<Idx>::ConstView;

private:
  template <std::size_t Idx>
  using Leaf = portablecollection::CollectionLeaf<Idx, Layout<Idx>>;

  template <std::size_t Idx>
  Leaf<Idx>& get() {
    return static_cast<Leaf<Idx>&>(impl_);
  }

  template <std::size_t Idx>
  Leaf<Idx> const& get() const {
    return static_cast<Leaf<Idx> const&>(impl_);
  }

  template <typename T>
  Leaf<index_t_<T>>& get() {
    return static_cast<Leaf<index_t_<T>>&>(impl_);
  }

  template <typename T>
  Leaf<index_t_<T>> const& get() const {
    return static_cast<Leaf<index_t_<T>> const&>(impl_);
  }

  static int32_t computeDataSize(const SizesArray& sizes) {
    int32_t ret = 0;
    constexpr_for<0, members_>([&sizes, &ret](auto i) { ret += Layout<i>::computeDataSize(sizes[i]); });
    return ret;
  }

public:
  PortableDeviceCollection() = default;

  PortableDeviceCollection(int32_t elements, TDev const& device)
      // allocate device memory
      : buffer_{cms::alpakatools::make_device_buffer<std::byte[]>(device, Layout<>::computeDataSize(elements))},
        impl_{buffer_->data(), elements} {
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout<>::alignment == 0);
    static_assert(members_ == 1);
  }

  template <typename TQueue, typename = std::enable_if_t<cms::alpakatools::is_queue_v<TQueue>>>
  PortableDeviceCollection(int32_t elements, TQueue const& queue)
      // allocate device memory asynchronously on the given work queue
      : buffer_{cms::alpakatools::make_device_buffer<std::byte[]>(queue, Layout<>::computeDataSize(elements))},
        impl_{buffer_->data(), elements} {
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout<>::alignment == 0);
    static_assert(members_ == 1);
  }

  PortableDeviceCollection(const SizesArray& sizes, TDev const& device)
      // allocate device memory
      : buffer_{cms::alpakatools::make_device_buffer<std::byte[]>(device, computeDataSize(sizes))},
        impl_{buffer_->data(), sizes} {
    constexpr_for<0, members_>(
        [&](auto i) { assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout<i>::alignment == 0); });
    constexpr auto alignment = Layout<0>::alignment;
    constexpr_for<1, members_>([&alignment](auto i) { static_assert(alignment == Layout<i>::alignment); });
  }

  template <typename TQueue, typename = std::enable_if_t<cms::alpakatools::is_queue_v<TQueue>>>
  PortableDeviceCollection(const SizesArray& sizes, TQueue const& queue)
      // allocate device memory asynchronously on the given work queue
      : buffer_{cms::alpakatools::make_device_buffer<std::byte[]>(queue, computeDataSize(sizes))},
        impl_{buffer_->data(), sizes} {
    constexpr_for<0, members_>(
        [&](auto i) { assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout<i>::alignment == 0); });
    constexpr auto alignment = Layout<0>::alignment;
    constexpr_for<1, members_>([&alignment](auto i) { static_assert(alignment == Layout<i>::alignment); });
  }

  // non-copyable
  PortableDeviceCollection(PortableDeviceCollection const&) = delete;
  PortableDeviceCollection& operator=(PortableDeviceCollection const&) = delete;

  // movable
  PortableDeviceCollection(PortableDeviceCollection&&) = default;
  PortableDeviceCollection& operator=(PortableDeviceCollection&&) = default;

  // default destructor
  ~PortableDeviceCollection() = default;

  // access the View by index
  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  View<Idx>& view() {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  ConstView<Idx> const& view() const {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  ConstView<Idx> const& const_view() const {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  View<Idx>& operator*() {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  ConstView<Idx> const& operator*() const {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  View<Idx>* operator->() {
    return &get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  ConstView<Idx> const* operator->() const {
    return &get<Idx>().view_;
  }

  // access the View by type
  template <typename T>
  typename T::View& view() {
    return get<T>().view_;
  }

  template <typename T>
  typename T::ConstView const& view() const {
    return get<T>().view_;
  }

  template <typename T>
  typename T::ConstView const& const_view() const {
    return get<T>().view_;
  }

  template <typename T>
  typename T::View& operator*() {
    return get<T>().view_;
  }

  template <typename T>
  typename T::ConstView const& operator*() const {
    return get<T>().view_;
  }

  template <typename T>
  typename T::View* operator->() {
    return &get<T>().view_;
  }

  template <typename T>
  typename T::ConstView const* operator->() const {
    return &get<T>().view_;
  }

  // access the Buffer
  Buffer buffer() { return *buffer_; }
  ConstBuffer buffer() const { return *buffer_; }
  ConstBuffer const_buffer() const { return *buffer_; }

  // Extract the sizes array
  SizesArray sizes() const {
    SizesArray ret;
    constexpr_for<0, members_>([&](auto i) { ret[i] = get<i>().layout_.metadata().size(); });
    return ret;
  }

private:
  std::optional<Buffer> buffer_;  //!
  Implementation impl_;           // (serialized: this is where the layouts live)
};

#endif  // DataFormats_Portable_interface_PortableDeviceCollection_h
