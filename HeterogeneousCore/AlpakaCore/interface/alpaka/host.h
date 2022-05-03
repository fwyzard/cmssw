#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_host_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_host_h

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/config.h"

// alpaka host device
inline const alpaka_common::DevHost host = alpaka::getDevByIdx<alpaka_common::PltfHost>(0u);

#endif  // HeterogeneousCore_AlpakaCore_interface_alpaka_host_h
