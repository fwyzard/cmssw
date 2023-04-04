#ifndef CondFormats_SiPixelObjects_SiPixelMappingHost_h
#define CondFormats_SiPixelObjects_SiPixelMappingHost_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelMappingLayout.h"

using SiPixelMappingHost = PortableHostCollection<SiPixelMappingLayout<>>;

#endif  // CondFormats_SiPixelObjects_SiPixelMappingHost_h
