#ifndef DataFormats_SiPixelDigiSoA_interface_alpaka_SiPixelDigiErrorsUtilities_h
#define DataFormats_SiPixelDigiSoA_interface_alpaka_SiPixelDigiErrorsUtilities_h

#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsLayout.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "HeterogeneousCore/AlpakaInterface/interface/SimpleVector.h"

// ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr SiPixelErrorCompactVec* error(
//     SiPixelDigiErrorsLayoutSoAView& errors) {
//   return (&errors.pixelErrorsVec());
// }
// ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr SiPixelErrorCompactVec const* error(
//     const SiPixelDigiErrorsLayoutSoAConstView& errors) {
//   return (&errors.pixelErrorsVec());
// }
// ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr SiPixelErrorCompact& error_data(
//     SiPixelDigiErrorsLayoutSoAView& errors) {
//   return (*errors.pixelErrors());
// }
// ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr SiPixelErrorCompact const& error_data(
//     const SiPixelDigiErrorsLayoutSoAConstView& errors) {
//   return (*errors.pixelErrors());
// }
// ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr SiPixelErrorCompactVec& error_vector(
//     SiPixelDigiErrorsLayoutSoAView& errors) {
//   return (errors.pixelErrorsVec());
// }

#endif  // DataFormats_SiPixelDigiSoA_interface_alpaka_SiPixelDigiErrorsUtilities_h