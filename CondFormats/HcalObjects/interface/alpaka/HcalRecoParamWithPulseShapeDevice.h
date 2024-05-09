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

#endif
