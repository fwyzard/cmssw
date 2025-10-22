#ifndef TrivialSerialisation_src_SerialiserFactoryPortable_h
#define TrivialSerialisation_src_SerialiserFactoryPortable_h

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "TrivialSerialisation/Common/interface/alpaka/Serialiser.h"
#include "TrivialSerialisation/Common/interface/alpaka/SerialiserBase.h"

#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace ngt {
    using SerialiserFactoryPortable = edmplugin::PluginFactory<ngt::SerialiserBase*()>;
  }

// Helper macro to define Serialiser plugins
#define COMMA ,
#define DEFINE_TRIVIAL_SERIALISER_PLUGIN(TYPE) \
  DEFINE_EDM_PLUGIN(ngt::SerialiserFactoryPortable, ngt::Serialiser<TYPE COMMA Device>, typeid(TYPE).name())

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif  // TrivialSerialisation_src_SerialiserFactoryPortable_h
