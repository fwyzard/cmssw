#ifndef TrivialSerialisation_src_SerialiserFactoryPortable_h
#define TrivialSerialisation_src_SerialiserFactoryPortable_h

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/SerialisationCore/interface/alpaka/Serialiser.h"
#include "HeterogeneousCore/SerialisationCore/interface/alpaka/SerialiserBase.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt {
  using SerialiserFactoryPortable = edmplugin::PluginFactory<SerialiserBase*()>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt

// Helper macro to define Serialiser plugins
#define DEFINE_PORTABLE_TRIVIAL_SERIALISER_PLUGIN(TYPE)                           \
  DEFINE_EDM_PLUGIN(ALPAKA_ACCELERATOR_NAMESPACE::ngt::SerialiserFactoryPortable, \
                    ALPAKA_ACCELERATOR_NAMESPACE::ngt::Serialiser<TYPE>,          \
                    typeid(TYPE).name())

#endif  // TrivialSerialisation_src_SerialiserFactoryPortable_h
