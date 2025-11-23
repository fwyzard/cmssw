#ifndef TrivialSerialisation_Common_interface_alpaka_Serialiser_h
#define TrivialSerialisation_Common_interface_alpaka_Serialiser_h

#include <memory>

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/WrapperBase.h"
#include "HeterogeneousCore/SerialisationCore/interface/alpaka/SerialiserBase.h"
#include "HeterogeneousCore/SerialisationCore/interface/alpaka/Reader.h"
#include "HeterogeneousCore/SerialisationCore/interface/alpaka/Writer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt {

  template <typename T>
  class Serialiser : public SerialiserBase {
  public:
    std::unique_ptr<WriterBase> writer() override { return std::make_unique<Writer<T>>(); }

    std::unique_ptr<const ReaderBase> reader(edm::WrapperBase const& wrapper) override {
      edm::Wrapper<T> const& w = dynamic_cast<edm::Wrapper<T> const&>(wrapper);
      return std::make_unique<Reader<T>>(w);
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt

#endif  // TrivialSerialisation_Common_interface_alpaka_Serialiser_h
