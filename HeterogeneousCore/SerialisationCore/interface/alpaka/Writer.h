#ifndef TrivialSerialisation_Common_interface_alpaka_Writer_h
#define TrivialSerialisation_Common_interface_alpaka_Writer_h

#include <cstddef>
#include <span>
#include <vector>

#include "HeterogeneousCore/SerialisationCore/interface/MemoryCopyTraits.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeDemangler.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/SerialisationCore/interface/AnyBuffer.h"
#include "HeterogeneousCore/SerialisationCore/interface/alpaka/WriterBase.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt {

  template <typename T>
  class Writer : public WriterBase {
    static_assert(::ngt::HasMemoryCopyTraits<T>, "No specialization of MemoryCopyTraits found for type T");

  public:
    using WrapperType = edm::Wrapper<T>;

    Writer() : WriterBase() {
      // See edm::Wrapper::construct_().
      if constexpr (requires { T(); }) {
        ptr_ = std::make_unique<edm::Wrapper<T>>(edm::WrapperBase::Emplace{});
      } else {
        ptr_ = std::make_unique<edm::Wrapper<T>>(edm::WrapperBase::Emplace{}, edm::kUninitialized);
      }
    }

    ::ngt::AnyBuffer uninitialized_parameters() const override {
      if constexpr (not ::ngt::HasTrivialCopyProperties<T>) {
        // if ::ngt::MemoryCopyTraits<T>::properties(...) is not declared, do not call it.
        return {};
      } else {
        // if ::ngt::MemoryCopyTraits<T>::properties(...) is declared, call it and wrap the result in an ::ngt::AnyBuffer
        return ::ngt::AnyBuffer(::ngt::MemoryCopyTraits<T>::properties(object()));
      }
    }

    void initialize(Queue& queue, ::ngt::AnyBuffer const& args) override {
      // XXX FIXME
      // if constexpr (not ::ngt::HasValidInitialize<T>) {
      if constexpr (false) {
        // If there is no valid initialize(), this shouldn't be present.
        static_assert(not ::ngt::HasTrivialCopyProperties<T>);
      } else if constexpr (not ::ngt::HasTrivialCopyProperties<T>) {
        // If T has no TrivialCopyProperties, call initialize() without any additional arguments.
        ::ngt::MemoryCopyTraits<T>::initialize(object(), queue);
      } else {
        // If T has TrivialCopyProperties, cast args to Properties and pass it as an additional argument to initialize().
        ::ngt::MemoryCopyTraits<T>::initialize(object(), queue, args.cast_to<::ngt::TrivialCopyProperties<T>>());
      }
    }

    std::vector<std::span<std::byte>> regions() override {
      static_assert(::ngt::HasRegions<T>);
      return ::ngt::MemoryCopyTraits<T>::regions(object());
    }

    void finalize() override {
      if constexpr (::ngt::HasTrivialCopyFinalize<T>) {
        ::ngt::MemoryCopyTraits<T>::finalize(object());
      }
    }

  private:
    const T& object() const { return static_cast<const WrapperType*>(ptr_.get())->bareProduct(); }

    T& object() { return static_cast<WrapperType*>(ptr_.get())->bareProduct(); }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt

#endif  // TrivialSerialisation_Common_interface_alpaka_Writer_h
