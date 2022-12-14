#ifndef DataFormats_Portable_interface_PortableHostCollection_h
#define DataFormats_Portable_interface_PortableHostCollection_h

#include <cassert>
#include <optional>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "DataFormats/Portable/interface/PortableCollectionCommon.h"

// generic SoA-based product in host memory
template <typename T0, typename T1 = void, typename T2 = void, typename T3 = void, typename T4 = void>
class PortableHostCollection {
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
  using Buffer = cms::alpakatools::host_buffer<std::byte[]>;
  using ConstBuffer = cms::alpakatools::const_host_buffer<std::byte[]>;
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
  portablecollection::CollectionLeaf<index_t_<T>, T>& get() {
    return static_cast<portablecollection::CollectionLeaf<index_t_<T>, T>&>(impl_);
  }

  template <typename T>
  const portablecollection::CollectionLeaf<index_t_<T>, T>& get() const {
    return static_cast<const portablecollection::CollectionLeaf<index_t_<T>, T>&>(impl_);
  }

  static int32_t computeDataSize(const std::array<int32_t, members_>& sizes) {
    int32_t ret = 0;
    constexpr_for<0, members_>([&sizes, &ret](auto i) { ret += Layout<i>::computeDataSize(sizes[i]); });
    return ret;
  }

public:
  PortableHostCollection() = default;

  PortableHostCollection(int32_t elements, alpaka_common::DevHost const& host)
      // allocate pageable host memory
      : buffer_{cms::alpakatools::make_host_buffer<std::byte[]>(Layout<>::computeDataSize(elements))},
        impl_{buffer_->data(), elements} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout<>::alignment == 0);
    static_assert(members_ == 1);
  }

  template <typename TQueue, typename = std::enable_if_t<cms::alpakatools::is_queue_v<TQueue>>>
  PortableHostCollection(int32_t elements, TQueue const& queue)
      // allocate pinned host memory associated to the given work queue, accessible by the queue's device
      : buffer_{cms::alpakatools::make_host_buffer<std::byte[]>(queue, Layout<>::computeDataSize(elements))},
        impl_{buffer_->data(), elements} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout<>::alignment == 0);
    static_assert(members_ == 1);
  }

  PortableHostCollection(const std::array<int32_t, members_>& sizes, alpaka_common::DevHost const& host)
      // allocate pinned host memory associated to the given work queue, accessible by the queue's device
      : buffer_{cms::alpakatools::make_host_buffer<std::byte[]>(computeDataSize(sizes))}, impl_{buffer_->data(), sizes} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    constexpr_for<0, members_>(
        [&](auto i) { assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout<i>::alignment == 0); });
    constexpr auto alignment = Layout<0>::alignment;
    constexpr_for<1, members_>([&alignment](auto i) { static_assert(alignment == Layout<i>::alignment); });
  }

  template <typename TQueue, typename = std::enable_if_t<cms::alpakatools::is_queue_v<TQueue>>>
  PortableHostCollection(const std::array<int32_t, members_>& sizes, TQueue const& queue)
      // allocate pinned host memory associated to the given work queue, accessible by the queue's device
      : buffer_{cms::alpakatools::make_host_buffer<std::byte[]>(queue, computeDataSize(sizes))},
        impl_{buffer_->data(), sizes} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    constexpr_for<0, members_>(
        [&](auto i) { assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout<i>::alignment == 0); });
    constexpr auto alignment = Layout<0>::alignment;
    constexpr_for<1, members_>([&alignment](auto i) { static_assert(alignment == Layout<i>::alignment); });
  }

  // non-copyable
  PortableHostCollection(PortableHostCollection const&) = delete;
  PortableHostCollection& operator=(PortableHostCollection const&) = delete;

  // movable
  PortableHostCollection(PortableHostCollection&&) = default;
  PortableHostCollection& operator=(PortableHostCollection&&) = default;

  // default destructor
  ~PortableHostCollection() = default;

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

  // part of the ROOT read streamer
  static void ROOTReadStreamer(PortableHostCollection* newObj, Implementation const& impl) {
    newObj->~PortableHostCollection();
    // use the global "host" object returned by cms::alpakatools::host()
    std::array<int32_t, members_> sizes;
    constexpr_for<0, members_>(
        [&sizes, &impl](auto i) { sizes[i] = static_cast<Leaf<i> const&>(impl).layout_.metadata().size(); });
    new (newObj) PortableHostCollection(sizes, cms::alpakatools::host());
    constexpr_for<0, members_>([&sizes, &newObj, &impl](auto i) {
      static_cast<Leaf<i>&>(newObj->impl_).layout_.ROOTReadStreamer(static_cast<Leaf<i> const&>(impl).layout_);
    });
  }

private:
  std::optional<Buffer> buffer_;  //!
  Implementation impl_;           // (serialized: this is where the layouts live)
};

#endif  // DataFormats_Portable_interface_PortableHostCollection_h
