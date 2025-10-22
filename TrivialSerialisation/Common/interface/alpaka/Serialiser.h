#ifndef TrivialSerialisation_Common_SerialiserPortable_h
#define TrivialSerialisation_Common_SerialiserPortable_h

#include "TrivialSerialisation/Common/interface/alpaka/SerialiserBase.h"
#include "TrivialSerialisation/Common/interface/alpaka/TrivialSerialiser.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "alpaka/dev/Traits.hpp"
namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace ngt {
    template <typename T, typename TDev = alpaka::DevCpu>
    class Serialiser : public SerialiserBase {
    public:
      std::unique_ptr<TrivialSerialiserBase> initialize(edm::WrapperBase& wrapper) override {
        edm::Wrapper<T>& w = dynamic_cast<edm::Wrapper<T>&>(wrapper);
        w.markAsPresent();
        return std::make_unique<TrivialSerialiser<T, TDev>>(w);
      }
      std::unique_ptr<const TrivialSerialiserBase> initialize(edm::WrapperBase const& wrapper) override {
        edm::Wrapper<T> const& w = dynamic_cast<edm::Wrapper<T> const&>(wrapper);
        return std::make_unique<const TrivialSerialiser<T, TDev>>(w);
      }
    };

  }  // namespace ngt

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // TrivialSerialisation_Common_SerialiserPortable_h
