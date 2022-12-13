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
  static constexpr size_t typeCount = CollectionTypeCount<T, T0, T1, T2, T3, T4>;

  static constexpr size_t membersCount = CollectionMembersCount<T0, T1, T2, T3, T4>;

public:
  using Buffer = cms::alpakatools::host_buffer<std::byte[]>;
  using ConstBuffer = cms::alpakatools::const_host_buffer<std::byte[]>;
  using Implementation = CollectionImpl<0, T0, T1, T2, T3, T4>;
  using TypeResolver = CollectionTypeResolver<T0, T1, T2, T3, T4>;
  using IdxResolver = CollectionIdxResolver<T0, T1, T2, T3, T4>;
  using SizesArray = std::array<int32_t, membersCount>;

  template <std::size_t Idx = 0, typename = std::enable_if_t<(membersCount > Idx)>>
  using Layout = typename TypeResolver::template Resolver<Idx>::type;
  template <std::size_t Idx = 0, typename = std::enable_if_t<(membersCount > Idx)>>
  using View = typename Layout<Idx>::View;
  template <std::size_t Idx = 0, typename = std::enable_if_t<(membersCount > Idx)>>
  using ConstView = typename Layout<Idx>::ConstView;

private:
  template <std::size_t Idx>
  CollectionLeaf<Idx, typename TypeResolver::template Resolver<Idx>::type>& get() {
    return dynamic_cast<CollectionLeaf<Idx, typename TypeResolver::template Resolver<Idx>::type>&>(impl_);
  }

  template <std::size_t Idx>
  const CollectionLeaf<Idx, typename TypeResolver::template Resolver<Idx>::type>& get() const {
    return dynamic_cast<const CollectionLeaf<Idx, typename TypeResolver::template Resolver<Idx>::type>&>(impl_);
  }

  template <typename T>
  CollectionLeaf<IdxResolver::template Resolver<T>::Idx, T>& get() {
    return dynamic_cast<CollectionLeaf<IdxResolver::template Resolver<T>::Idx, T>&>(impl_);
  }

  template <typename T>
  const CollectionLeaf<IdxResolver::template Resolver<T>::Idx, T>& get() const {
    return dynamic_cast<const CollectionLeaf<IdxResolver::template Resolver<T>::Idx, T>&>(impl_);
  }

public:
  PortableHostCollection() = default;

  PortableHostCollection(int32_t elements, alpaka_common::DevHost const& host)
      // allocate pageable host memory
      : buffer_{cms::alpakatools::make_host_buffer<std::byte[]>(Layout<>::computeDataSize(elements))},
        impl_{buffer_->data(), elements} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout<>::alignment == 0);
    static_assert(membersCount == 1);
  }

  template <typename TQueue, typename = std::enable_if_t<cms::alpakatools::is_queue_v<TQueue>>>
  PortableHostCollection(int32_t elements, TQueue const& queue)
      // allocate pinned host memory associated to the given work queue, accessible by the queue's device
      : buffer_{cms::alpakatools::make_host_buffer<std::byte[]>(queue, Layout<>::computeDataSize(elements))},
        impl_{buffer_->data(), elements} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout<>::alignment == 0);
    static_assert(membersCount == 1);
  }

  static int32_t computeDataSize(const std::array<int32_t, membersCount>& sizes) {
    int32_t ret = 0;
    constexpr_for<0, membersCount, 1>(
        [&sizes, &ret](auto i) { ret += TypeResolver::template Resolver<i>::type::computeDataSize(sizes[i]); });
    return ret;
  }

  PortableHostCollection(const std::array<int32_t, membersCount>& sizes, alpaka_common::DevHost const& host)
      // allocate pinned host memory associated to the given work queue, accessible by the queue's device
      : buffer_{cms::alpakatools::make_host_buffer<std::byte[]>(computeDataSize(sizes))}, impl_{buffer_->data(), sizes} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    constexpr_for<0, membersCount, 1>(
        [&](auto i) { assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout<i>::alignment == 0); });
    constexpr auto alignment = Layout<0>::alignment;
    constexpr_for<1, membersCount, 1>([&alignment](auto i) { static_assert(alignment == Layout<i>::alignment); });
  }

  template <typename TQueue, typename = std::enable_if_t<cms::alpakatools::is_queue_v<TQueue>>>
  PortableHostCollection(const std::array<int32_t, membersCount>& sizes, TQueue const& queue)
      // allocate pinned host memory associated to the given work queue, accessible by the queue's device
      : buffer_{cms::alpakatools::make_host_buffer<std::byte[]>(queue, computeDataSize(sizes))},
        impl_{buffer_->data(), sizes} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    constexpr_for<0, membersCount, 1>(
        [&](auto i) { assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout<i>::alignment == 0); });
    constexpr auto alignment = Layout<0>::alignment;
    constexpr_for<1, membersCount, 1>([&alignment](auto i) { static_assert(alignment == Layout<i>::alignment); });
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
  template <std::size_t Idx = 0, typename = std::enable_if_t<(membersCount > Idx)>>
  View<Idx>& view() {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(membersCount > Idx)>>
  ConstView<Idx> const& view() const {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(membersCount > Idx)>>
  ConstView<Idx> const& const_view() const {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(membersCount > Idx)>>
  View<Idx>& operator*() {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(membersCount > Idx)>>
  ConstView<Idx> const& operator*() const {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(membersCount > Idx)>>
  View<Idx>* operator->() {
    return &get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(membersCount > Idx)>>
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
    constexpr_for<0, membersCount, 1>([&](auto i) { ret[i] = get<i>().layout_.metadata().size(); });
    return ret;
  }

  // part of the ROOT read streamer
  static void ROOTReadStreamer(PortableHostCollection* newObj, Implementation const& impl) {
    newObj->~PortableHostCollection();
    // use the global "host" object returned by cms::alpakatools::host()
    std::array<int32_t, membersCount> sizes;
    constexpr_for<0, membersCount, 1>([&sizes, &impl](auto i) {
      sizes[i] = impl.CollectionLeaf<i, typename TypeResolver::template Resolver<i>::type>::layout_.metadata().size();
    });
    new (newObj) PortableHostCollection(sizes, cms::alpakatools::host());
    constexpr_for<0, membersCount, 1>([&sizes, &newObj, &impl](auto i) {
      newObj->impl_.CollectionLeaf<i, typename TypeResolver::template Resolver<i>::type>::layout_.ROOTReadStreamer(
          impl.CollectionLeaf<i, typename TypeResolver::template Resolver<i>::type>::layout_);
    });
  }

private:
  std::optional<Buffer> buffer_;  //!
  Implementation impl_;           // (serialized: this is where the layouts live)
};

#endif  // DataFormats_Portable_interface_PortableHostCollection_h
