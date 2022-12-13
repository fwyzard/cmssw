#ifndef DataFormats_Portable_interface_PortableCollectionCommon_h
#define DataFormats_Portable_interface_PortableCollectionCommon_h

#include <cstddef>

template <std::size_t Start, std::size_t End, std::size_t Inc = 1, typename F>
constexpr void constexpr_for(F&& f) {
  if constexpr (Start < End) {
    f(std::integral_constant<std::size_t, Start>());
    constexpr_for<Start + Inc, End, Inc>(f);
  }
}

template <std::size_t Idx, typename T>
struct CollectionLeaf {
  CollectionLeaf() = default;
  CollectionLeaf(std::byte* buffer, int32_t elements) : layout_(buffer, elements), view_(layout_) {}
  template <std::size_t N>
  CollectionLeaf(std::byte* buffer, std::array<int32_t, N> const& sizes) : layout_(buffer, sizes[Idx]), view_(layout_) {
    static_assert(N > Idx);
  }
  using Layout = T;
  using View = typename Layout::View;
  using ConstView = typename Layout::ConstView;
  Layout layout_;  //
  View view_;      //!
};

template <std::size_t Idx, typename T, typename... Args>
struct CollectionImpl : public CollectionLeaf<Idx, T>, public CollectionImpl<Idx + 1, Args..., void> {
  CollectionImpl() = default;
  CollectionImpl(std::byte* buffer, int32_t elements) : CollectionLeaf<Idx, T>(buffer, elements) {}

  template <std::size_t N>
  CollectionImpl(std::byte* buffer, std::array<int32_t, N> const& sizes)
      : CollectionLeaf<Idx, T>(buffer, sizes),
        CollectionImpl<Idx + 1, Args..., void>(CollectionLeaf<Idx, T>::layout_.metadata().nextByte(), sizes) {}
};

template <std::size_t Idx>
struct CollectionImpl<Idx, void, void, void, void, void> {
  CollectionImpl() = default;
  template <std::size_t N>
  CollectionImpl(std::byte* buffer, std::array<int32_t, N> const& sizes) {
    static_assert(N == Idx);
  }
};

template <std::size_t Idx, typename... Args>
using CollectionTypeResolver = typename std::tuple_element<Idx, std::tuple<Args...>>::type;

template <typename T, typename... Args>
static constexpr std::size_t CollectionTypeCount = ((std::is_same<T, Args>::value ? 1 : 0) + ...);

template <typename... Args>
static constexpr std::size_t CollectionMembersCount = sizeof...(Args) - CollectionTypeCount<void, Args...>;

template <typename T, typename Tuple>
struct CollectionIdxResolverImpl;

template <typename T, typename... Args>
struct CollectionIdxResolverImpl<T, std::tuple<T, Args...>> {
  static_assert(CollectionTypeCount<T, Args...> == 0, "the requested type appears more than once among the arguments");
  static const std::size_t value = 0;
};

template <typename T, typename U, typename... Args>
struct CollectionIdxResolverImpl<T, std::tuple<U, Args...>> {
  static_assert(not std::is_same_v<T, U>);
  static_assert(CollectionTypeCount<T, Args...> == 1, "the requested type does not appear among the arguments");
  static const std::size_t value = 1 + CollectionIdxResolverImpl<T, std::tuple<Args...>>::value;
};

template <typename T, typename... Args>
static constexpr std::size_t CollectionIdxResolver = CollectionIdxResolverImpl<T, std::tuple<Args...>>::value;

// TODO: namespace this

#endif  // DataFormats_Portable_interface_PortableCollectionCommon_h
