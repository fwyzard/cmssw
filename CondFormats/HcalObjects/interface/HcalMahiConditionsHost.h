#ifndef CondFormats_HcalObjects_interface_HcalMahiConditionsHost_h
#define CondFormats_HcalObjects_interface_HcalMahiConditionsHost_h

#include "CondFormats/HcalObjects/interface/HcalMahiConditionsSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

using HcalMahiConditionsPortableHost = PortableHostCollection<HcalMahiConditionsSoA>;

#endif
