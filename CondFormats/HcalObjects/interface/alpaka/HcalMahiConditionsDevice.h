#ifndef CondFormats_HcalObjects_interface_alpaka_HcalMahiConditionsDevice_h
#define CondFormats_HcalObjects_interface_alpaka_HcalMahiConditionsDevice_h

#include "CondFormats/HcalObjects/interface/HcalMahiConditionsHost.h"
#include "CondFormats/HcalObjects/interface/HcalMahiConditionsSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using ::HcalMahiConditionsPortableHost;
  using HcalMahiConditionsPortableDevice = PortableCollection<HcalMahiConditionsSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
