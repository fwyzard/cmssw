#ifndef CondFormats_SiPixelObjects_alpaka_SiPixelGainCalibrationForHLTDevice_h
#define CondFormats_SiPixelObjects_alpaka_SiPixelGainCalibrationForHLTDevice_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLTLayout.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using SiPixelGainCalibrationForHLTDevice = PortableCollection<SiPixelGainCalibrationForHLTLayout<>>;
  using SiPixelGainCalibrationForHLTHost = PortableHostCollection<SiPixelGainCalibrationForHLTLayout<>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif  // CondFormats_SiPixelObjects_alpaka_SiPixelGainCalibrationForHLTDevice_h
