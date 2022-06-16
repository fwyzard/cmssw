#ifndef DataFormats_XyzId_interface_XyzIdSoA_h
#define DataFormats_XyzId_interface_XyzIdSoA_h

#include "DataFormats/XyzId/interface/SoACommon.h"
#include "DataFormats/XyzId/interface/SoALayout.h"
#include "DataFormats/XyzId/interface/SoAView.h"



#if 1
GENERATE_SOA_LAYOUT(XyzIdSoALayout, 
                             /*XyzIdSoAView,*/
                             // columns: one value per element
                             SOA_COLUMN(double, x),  
                             SOA_COLUMN(double, y),
                             SOA_COLUMN(double, z),
                             SOA_COLUMN(int32_t, id))
#endif
using XyzIdSoA = XyzIdSoALayout<>;

#endif  // DataFormats_XyzId_interface_XyzIdSoA_h
