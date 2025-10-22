#ifndef TrivialSerialisation_Common_SerialiserBasePortable_h
#define TrivialSerialisation_Common_SerialiserBasePortable_h

#include "DataFormats/Common/interface/WrapperBase.h"
#include "TrivialSerialisation/Common/interface/alpaka/TrivialSerialiserBase.h"


#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
namespace ALPAKA_ACCELERATOR_NAMESPACE {

namespace ngt {
  class SerialiserBase {
  public:
    SerialiserBase() = default;

    virtual std::unique_ptr<TrivialSerialiserBase> initialize(edm::WrapperBase& wrapper) = 0;
    virtual std::unique_ptr<const TrivialSerialiserBase> initialize(const edm::WrapperBase& wrapper) = 0;

    virtual ~SerialiserBase() = default;
  };
}  // namespace ngt
}

#endif  // TrivialSerialisation_Common_SerialiserBasePortable_h
