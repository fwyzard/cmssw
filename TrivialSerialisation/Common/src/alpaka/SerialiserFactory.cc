

#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"



#include "TrivialSerialisation/Common/interface/alpaka/SerialiserFactory.h"


EDM_REGISTER_PLUGINFACTORY(ALPAKA_ACCELERATOR_NAMESPACE::ngt::SerialiserFactoryPortable, "SerialiserFactoryPortable");
