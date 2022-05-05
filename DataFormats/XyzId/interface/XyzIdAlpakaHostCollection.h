#ifndef DataFormats_XyzId_interface_XyzIdAlpakaHostCollection_h
#define DataFormats_XyzId_interface_XyzIdAlpakaHostCollection_h

#include "DataFormats/Portable/interface/AlpakaHostCollection.h"
#include "DataFormats/XyzId/interface/XyzIdSoA.h"

// SoA with x, y, z, id fields in pinned host memory
using XyzIdAlpakaHostCollection = AlpakaHostCollection<XyzIdSoA>;

//class XyzIdAlpakaHostCollection : public AlpakaHostCollection<XyzIdSoA> {
//  using AlpakaHostCollection<XyzIdSoA>::AlpakaHostCollection;
//};

#endif  // DataFormats_XyzId_interface_XyzIdAlpakaHostCollection_h
