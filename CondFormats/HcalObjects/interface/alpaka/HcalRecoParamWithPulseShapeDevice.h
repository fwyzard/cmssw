#ifndef CondFormats_HcalObjects_interface_alpaka_HcalRecoParamWithPulseShapeDevice_h
#define CondFormats_HcalObjects_interface_alpaka_HcalRecoParamWithPulseShapeDevice_h

#include "CondFormats/HcalObjects/interface/HcalRecoParamWithPulseShapeHost.h"
#include "CondFormats/HcalObjects/interface/HcalRecoParamWithPulseShapeSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using HcalRecoParamWithPulseShapeDevice = HcalRecoParamWithPulseShapeT<Device>;
}

namespace cms::alpakatools {
  template <>
  struct CopyToDevice<HcalRecoParamWithPulseShapeHost> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, HcalRecoParamWithPulseShapeHost const& hostProduct) {
      using RecoParamCopy = CopyToDevice<HcalRecoParamWithPulseShapeHost::RecoParamCollection>;
      using PulseShapeCopy = CopyToDevice<HcalRecoParamWithPulseShapeHost::PulseShapeCollection>;
      //using TDevice = typename alpaka::trait::DevType<TQueue>::type;
      using TDevice = alpaka::Dev<TQueue>;
      return HcalRecoParamWithPulseShapeT<TDevice>(RecoParamCopy::copyAsync(queue, hostProduct.recoParam()),
                                                   PulseShapeCopy::copyAsync(queue, hostProduct.pulseShape()));
    }
  };
}  // namespace cms::alpakatools

#endif
