#ifndef TrivialSerialisation_Common_TrivialSerialiserPortable_h
#define TrivialSerialisation_Common_TrivialSerialiserPortable_h

#include <cstdio>
#include <vector>
#include <span>
#include <cstddef>
#include <type_traits>
#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/AnyBuffer.h"
#include "DataFormats/Common/interface/TrivialCopyTraits.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "TrivialSerialisation/Common/interface/alpaka/TrivialSerialiserBase.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeDemangler.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "alpaka/dev/Traits.hpp"
// defines all methods of TrivialSerialiserBase

namespace ALPAKA_ACCELERATOR_NAMESPACE {

namespace ngt {

  template <typename T, typename TQueue>
  class TrivialSerialiser : public TrivialSerialiserBase {
    static_assert(edm::HasTrivialCopyTraits<T>, "No specialization of TrivialCopyTraits found for type T");

  public:
    using WrapperType = edm::Wrapper<T>;
    TrivialSerialiser(WrapperType const& obj) : TrivialSerialiserBase(&obj) {}

    void initialize(edm::AnyBuffer const& args, const Queue& queue) override;
    edm::AnyBuffer parameters() const override;
    std::vector<std::span<const std::byte>> regions() const override;
    std::vector<std::span<std::byte>> regions() override;
    void trivialCopyFinalize() override;

  private:
    const T& getWrappedObj_() const {
      WrapperType const& w = static_cast<WrapperType const&>(*getWrapperBasePtr());
      if (not w.isPresent()) {
        throw edm::Exception(edm::errors::LogicError) << "Attempt to access an empty Wrapper";
      }
      // return *w.product();
      return w.bareProduct();
    }

    T& getWrappedObj_() {
      WrapperType& w = const_cast<WrapperType&>(static_cast<WrapperType const&>(*getWrapperBasePtr()));
      if (not w.isPresent()) {
        throw edm::Exception(edm::errors::LogicError) << "Attempt to access an empty Wrapper";
      }
      return w.bareProduct();
    }
  };

  template <typename T, typename TQueue>
  void TrivialSerialiser<T, TQueue>::initialize(edm::AnyBuffer const& args, const Queue& queue) {
    // if constexpr (not edm::HasValidInitialize<T>) {
    if constexpr (0) {
      // If there is no valid initialize(), there shouldn't be present
      static_assert(not edm::HasTrivialCopyProperties<T>);
      return;
    } else {
      printf("QUEUE TYPE IS %s\n", typeid(Queue).name());

      auto& w = static_cast<WrapperType const&>(*getWrapperBasePtr());
      // Each serialiser stores a pointer to the wrapper used to initialize it as "const edm::WrapperBase*"
      // For serialisers used for writing, that were initialized with a non-const wrapper, the const_cast below is safe because the wrapper was originally non-const.
      // For serialisers used for reading, that were initialized with a const wrapper, this function cannot be called because it is not marked as const.
      if constexpr (not edm::HasTrivialCopyProperties<T>) {
        // if T has no TrivialCopyProperties, call initialize() without any additional arguments
        edm::TrivialCopyTraits<T>::initialize(const_cast<WrapperType&>(w).bareProduct(), queue);
      } else {
        // if T has TrivialCopyProperties, cast args to Properties and pass it as an additional argument to initialize()
        edm::TrivialCopyTraits<T>::initialize(
            const_cast<WrapperType&>(w).bareProduct(), queue, args.cast_to<edm::TrivialCopyProperties<T>>());
      }
    }
  }

  template <typename T, typename TDev>
  inline edm::AnyBuffer TrivialSerialiser<T, TDev>::parameters() const {
    const T& obj = getWrappedObj_();
    if constexpr (not edm::HasTrivialCopyProperties<T>) {
      // if edm::TrivialCopyTraits<T>::properties(...) is not declared, do not call it.
      return {};
    } else {
      // if edm::TrivialCopyTraits<T>::properties(...) is declared, call it and wrap the result in an edm::AnyBuffer
      edm::TrivialCopyProperties<T> p = edm::TrivialCopyTraits<T>::properties(obj);
      return edm::AnyBuffer(p);
    }
  }

  template <typename T, typename TDev>
  inline std::vector<std::span<const std::byte>> TrivialSerialiser<T, TDev>::regions() const {
    static_assert(edm::HasRegions<T>);
    const T& obj = getWrappedObj_();
    return edm::TrivialCopyTraits<T>::regions(obj);
  }

  template <typename T, typename TDev>
  inline std::vector<std::span<std::byte>> TrivialSerialiser<T, TDev>::regions() {
    static_assert(edm::HasRegions<T>);
    T& obj = getWrappedObj_();
    return edm::TrivialCopyTraits<T>::regions(obj);
  }

  template <typename T, typename TDev>
  inline void TrivialSerialiser<T, TDev>::trivialCopyFinalize() {
    if constexpr (edm::HasTrivialCopyFinalize<T>) {
      const T& obj = getWrappedObj_();
      edm::TrivialCopyTraits<T>::finalize(obj);
    }
  }

}  // namespace ngt

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // TrivialSerialisation_Common_TrivialSerialiserPortable_h
