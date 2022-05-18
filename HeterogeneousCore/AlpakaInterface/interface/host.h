#ifndef HeterogeneousCore_AlpakaInterface_interface_host_h
#define HeterogeneousCore_AlpakaInterface_interface_host_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// alpaka host device
inline const alpaka_common::DevHost host = alpaka::getDevByIdx<alpaka_common::PltfHost>(0u);

#endif  // HeterogeneousCore_AlpakaInterface_interface_host_h
