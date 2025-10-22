#ifndef TrivialSerialisation_Common_interface_TrivialSerialiserBasePortable_h
#define TrivialSerialisation_Common_interface_TrivialSerialiserBasePortable_h

#include "DataFormats/Common/interface/AnyBuffer.h"
#include "DataFormats/Common/interface/WrapperBase.h"

#include <span>
#include <vector>

#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
// #include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"


// #ifndef ALPAKA_ACCELERATOR_NAMESPACE
//   #define ALPAKA_ACCELERATOR_NAMESPACE alpaka_serial_sync
//   namespace alpaka_serial_sync {
//     using Device = alpaka::DevCpu;
//   }
// #endif


namespace ALPAKA_ACCELERATOR_NAMESPACE {


namespace ngt {
  class TrivialSerialiserBase {
  public:
    TrivialSerialiserBase(const edm::WrapperBase* ptr) : ptr_(ptr) {}

    virtual void initialize(edm::AnyBuffer const& args, const Queue& device) = 0;
    virtual edm::AnyBuffer parameters() const = 0;
    virtual std::vector<std::span<const std::byte>> regions() const = 0;
    virtual std::vector<std::span<std::byte>> regions() = 0;
    virtual void trivialCopyFinalize() = 0;

    const edm::WrapperBase* getWrapperBasePtr() const { return ptr_; }


    virtual ~TrivialSerialiserBase() = default;

  private:
    const edm::WrapperBase* ptr_;
  };

}  // namespace ngt

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // TrivialSerialisation_Common_interface_TrivialSerialiserBasePortable_h
