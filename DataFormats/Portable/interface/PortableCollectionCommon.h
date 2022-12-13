#ifndef DataFormats_Portable_interface_PortableCollectionCommon_h
#define DataFormats_Portable_interface_PortableCollectionCommon_h

#include <cstddef>

template <std::size_t Start, std::size_t End, std::size_t Inc = 1, class F>
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
  CollectionLeaf(std::byte* buffer, std::array<int32_t, N> sizes) : layout_(buffer, sizes[Idx]), view_(layout_) {
    static_assert(N > Idx);
  }
  using Layout = T;
  using View = typename Layout::View;
  using ConstView = typename Layout::ConstView;
  Layout layout_;  //
  View view_;      //!
};

template <std::size_t Idx, typename T0, typename T1, typename T2, typename T3, typename T4>
struct CollectionImpl : public CollectionLeaf<Idx, T0>, public CollectionImpl<Idx + 1, T1, T2, T3, T4, void> {
  CollectionImpl() = default;
  CollectionImpl(std::byte* buffer, int32_t elements) : CollectionLeaf<Idx, T0>(buffer, elements) {}

  template <size_t N>
  CollectionImpl(std::byte* buffer, std::array<int32_t, N> sizes)
      : CollectionLeaf<Idx, T0>(buffer, sizes),
        CollectionImpl<Idx + 1, T1, T2, T3, T4, void>(CollectionLeaf<Idx, T0>::layout_.metadata().nextByte(), sizes) {}
};

// This should work, but does not due to a GCC bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=85282
// Workaround implementation below
/*
template <typename T0, typename T1, typename T2, typename T3, typename T4>
struct CollectionTypeResolver{
  template <std::size_t Idx>
  struct Resolver{
    static_assert(Idx != 0);
    using type = typename CollectionTypeResolver<T1, T2, T3, T4, void>::template Resolver<Idx - 1>::type;
  };

  template <>
  struct Resolver<0>{
    using type = T0;
  };
};
*/

template <typename T0, typename T1, typename T2, typename T3, typename T4>
struct CollectionTypeResolver {
  template <std::size_t Idx, class = void>
  struct Resolver {
    static_assert(Idx != 0);
    using type = typename CollectionTypeResolver<T1, T2, T3, T4, void>::template Resolver<Idx - 1>::type;
  };

  template <std::size_t Idx>
  struct Resolver<Idx, std::enable_if_t<Idx == 0>> {
    using type = T0;
  };
};

template <typename T, typename T0, typename T1 = void, typename T2 = void, typename T3 = void, typename T4 = void>
static constexpr size_t CollectionTypeCount = (std::is_same<T0, T>::value ? 1 : 0) +
                                              (std::is_same<T1, T>::value ? 1 : 0) +
                                              (std::is_same<T2, T>::value ? 1 : 0) +
                                              (std::is_same<T3, T>::value ? 1 : 0) +
                                              (std::is_same<T4, T>::value ? 1 : 0);

template <typename T0, typename T1 = void, typename T2 = void, typename T3 = void, typename T4 = void>
static constexpr size_t CollectionMembersCount = (std::is_same<T0, void>::value ? 0 : 1) +
                                                 (std::is_same<T1, void>::value ? 0 : 1) +
                                                 (std::is_same<T2, void>::value ? 0 : 1) +
                                                 (std::is_same<T3, void>::value ? 0 : 1) +
                                                 (std::is_same<T4, void>::value ? 0 : 1);

template <typename T0, typename T1 = void, typename T2 = void, typename T3 = void, typename T4 = void>
struct CollectionIdxResolver {
  template <typename T, class = void>
  struct Resolver {
    static_assert(CollectionTypeCount<T, T0, T1, T2, T3, T4> == 1);
    static_assert(not std::is_same<T, T0>::value);
    static constexpr std::size_t Idx = 1 + CollectionIdxResolver<T1, T2, T3, T4, void>::template Resolver<T>::Idx;
  };

  template <typename T>
  struct Resolver<T, std::enable_if_t<std::is_same<T, T0>::value>> {
    static_assert(CollectionTypeCount<T, T0, T1, T2, T3, T4> == 1);
    static_assert(std::is_same<T, T0>::value);
    static constexpr std::size_t Idx = 0;
  };
};

template <std::size_t Idx>
struct CollectionImpl<Idx, void, void, void, void, void> {
  CollectionImpl() = default;
  template <size_t N>
  CollectionImpl(std::byte* buffer, std::array<int32_t, N> sizes) {
    static_assert(N == Idx);
  }
};

// TODO: namespace this

#endif  // DataFormats_Portable_interface_PortableCollectionCommon_h
