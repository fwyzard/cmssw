#pragma once

#include "DataFormats/Portable/interface/AlpakaHostCollection.h"
#include "DataFormats/XyzId/interface/XyzIdSoA.h"

// SoA with x, y, z, id fields in pinned host memory
using XyzIdAlpakaHostCollection = AlpakaHostCollection<XyzIdSoA>;
