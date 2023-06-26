#ifndef DataFormats_SiPixelDigi_interface_alpaka_SiPixelDigisCollection_h
#define DataFormats_SiPixelDigi_interface_alpaka_SiPixelDigisCollection_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisHost.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisLayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  using SiPixelDigisCollection = SiPixelDigisHost;
#else
  using SiPixelDigisCollection = SiPixelDigisDevice<Device>;

#endif
  using SiPixelDigisSoA = SiPixelDigisCollection;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <>
  struct CopyToHost<ALPAKA_ACCELERATOR_NAMESPACE::SiPixelDigisSoA> {
    template <typename TQueue>
    static auto copyAsync(TQueue &queue, ALPAKA_ACCELERATOR_NAMESPACE::SiPixelDigisSoA const &srcData) {
      SiPixelDigisHost dstData(srcData.view().metadata().size(), queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      dstData.setNModulesDigis(srcData.nModules(), srcData.nDigis());
      return dstData;
    }
  };
}  // namespace cms::alpakatools

// }  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif  // DataFormats_SiPixelDigi_interface_alpaka_SiPixelDigisCollection_h