#ifndef DataFormats_XyzId_interface_XyzIdHostCollection_h
#define DataFormats_XyzId_interface_XyzIdHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/XyzId/interface/XyzIdSoA.h"

// SoA with x, y, z, id fields in pinned host memory
using XyzIdHostCollection = PortableHostCollection<XyzIdSoA>;

#endif  // DataFormats_XyzId_interface_XyzIdHostCollection_h
