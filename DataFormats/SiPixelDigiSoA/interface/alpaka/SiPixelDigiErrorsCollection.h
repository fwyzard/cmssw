#ifndef DataFormats_SiPixelDigiSoA_interface_alpaka_SiPixelDigiErrorsCollection_h
#define DataFormats_SiPixelDigiSoA_interface_alpaka_SiPixelDigiErrorsCollection_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsHost.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigiErrorsUtilities.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "HeterogeneousCore/AlpakaInterface/interface/SimpleVector.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  using SiPixelDigiErrorsCollection = SiPixelDigiErrorsHost;
#else
  using SiPixelDigiErrorsCollection = SiPixelDigiErrorsDevice<Device>;
#endif
  using SiPixelDigiErrorsSoA = SiPixelDigiErrorsCollection;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <>
  struct CopyToHost<ALPAKA_ACCELERATOR_NAMESPACE::SiPixelDigiErrorsSoA> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, ALPAKA_ACCELERATOR_NAMESPACE::SiPixelDigiErrorsSoA const& srcData) {
      // auto error_vector_d = srcData.error_vector();
      // auto error_data_h = cms::alpakatools::make_host_buffer<SiPixelErrorCompact[]>(error_vector_d.capacity());
      // auto error_data_d = srcData.error_data();

      // if (not error_vector_d.empty()) {
      //   alpaka::memcpy(queue, error_data_h, error_data_d);
      // }

      // SiPixelDigiErrorsHost dstData(error_vector_d.capacity(), error_data_h);
      SiPixelDigiErrorsHost dstData(srcData.error_vector().capacity(), queue);
      // SiPixelDigiErrorsHost dstData(error_vector(srcData.view()).capacity(), queue);

      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());

      return dstData;
    }
  };
}  // namespace cms::alpakatools

#endif  // DataFormats_SiPixelDigiSoA_interface_alpaka_SiPixelDigiErrorsCollection_h