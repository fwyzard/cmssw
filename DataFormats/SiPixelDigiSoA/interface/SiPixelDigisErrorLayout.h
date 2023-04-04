#ifndef DataFormats_SiPixelDigi_SiPixelDigisErrorLayout_h
#define DataFormats_SiPixelDigi_SiPixelDigisErrorLayout_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"

GENERATE_SOA_LAYOUT(SiPixelDigisErrorLayout, SOA_COLUMN(SiPixelErrorCompact, pixelErrors))

using SiPixelDigisErrorLayoutSoA = SiPixelDigisErrorLayout<>;
using SiPixelDigisErrorLayoutSoAView = SiPixelDigisErrorLayout<>::View;
using SiPixelDigisErrorLayoutSoAConstView = SiPixelDigisErrorLayout<>::ConstView;

#endif  // DataFormats_SiPixelDigi_SiPixelDigisErrorLayout_h
