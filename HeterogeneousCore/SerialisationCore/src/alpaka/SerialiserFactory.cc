#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/SerialisationCore/interface/alpaka/SerialiserFactory.h"

EDM_REGISTER_PLUGINFACTORY(ALPAKA_ACCELERATOR_NAMESPACE::ngt::SerialiserFactoryPortable, "SerialiserFactoryPortable");
