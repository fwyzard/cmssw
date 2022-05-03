#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_macros_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_macros_h

#include "FWCore/Framework/interface/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/config.h"

// force expanding ALPAKA_ACCELERATOR_NAMESPACE before stringification inside DEFINE_FWK_MODULE
#define DEFINE_FWK_ALPAKA_MODULE2(name) DEFINE_FWK_MODULE(name)
#define DEFINE_FWK_ALPAKA_MODULE(name) DEFINE_FWK_ALPAKA_MODULE2(ALPAKA_ACCELERATOR_NAMESPACE::name)

#endif  // HeterogeneousCore_AlpakaCore_interface_alpaka_macros_h
