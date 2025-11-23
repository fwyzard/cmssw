

// #include <alpaka/alpaka.hpp>
// #include <Eigen/Core>
//
// #include "DataFormats/Common/interface/DeviceProduct.h"
// #include "DataFormats/Portable/interface/PortableCollection.h"
//
// #include "DataFormats/HcalRecHit/interface/HcalRecHitSoA.h"
// #include "DataFormats/ParticleFlowReco/interface/PFRecHitSoA.h"
// #include "DataFormats/ParticleFlowReco/interface/PFClusterSoA.h"
// #include "DataFormats/ParticleFlowReco/interface/PFRecHitFractionSoA.h"
// #include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHitSoA.h"
// #include "DataFormats/EcalDigi/interface/EcalDigiSoA.h"
//
// // #include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceObject.h"
// #include "DataFormats/Portable/interface/PortableDeviceObject.h"
// #include "DataFormats/PortableTestObjects/interface/TestStruct.h"
// #include "DataFormats/Portable/interface/PortableObject.h"
// #include "DataFormats/PortableTestObjects/interface/TestSoA.h"
// #include "HeterogeneousCore/AlpakaInterface/interface/config.h"
// #include "HeterogeneousCore/SerialisationCore/interface/alpaka/SerialiserFactory.h"
// #include "HeterogeneousCore/SerialisationCore/interface/alpaka/TrivialSerialiser.h"
// #include "HeterogeneousCore/SerialisationCore/interface/alpaka/SerialiserFactory.h"
// #include "HeterogeneousCore/SerialisationCore/interface/alpaka/Serialiser.h"
//
// using namespace ALPAKA_ACCELERATOR_NAMESPACE;
//
//

// using DeviceProductPortableCollectionHcalRecHitSoALayout = edm::DeviceProduct<PortableCollection<hcal::HcalRecHitSoALayout<128, false>, Device>>;
// DEFINE_EDM_PLUGIN(ngt::SerialiserFactory,
//                   ngt::Serialiser<DeviceProductPortableCollectionHcalRecHitSoALayout>,
//                   typeid(DeviceProductPortableCollectionHcalRecHitSoALayout).name());
//
// using PortableCollectionPFRecHitSoALayout = PortableCollection<reco::PFRecHitSoALayout<128, false>, Device>;
// DEFINE_EDM_PLUGIN(ngt::SerialiserFactory,
//                   ngt::Serialiser<PortableCollectionPFRecHitSoALayout>,
//                   typeid(PortableCollectionPFRecHitSoALayout).name());
//
// using PortableCollectionPFClusterSoALayout = PortableCollection<reco::PFClusterSoALayout<128, false>, Device>;
// DEFINE_EDM_PLUGIN(ngt::SerialiserFactory,
//                   ngt::Serialiser<PortableCollectionPFClusterSoALayout>,
//                   typeid(PortableCollectionPFClusterSoALayout).name());
//
// using PortableCollectionPFRecHitFractionSoALayout =
//     PortableCollection<reco::PFRecHitFractionSoALayout<128, false>, Device>;
// DEFINE_EDM_PLUGIN(ngt::SerialiserFactory,
//                   ngt::Serialiser<PortableCollectionPFRecHitFractionSoALayout>,
//                   typeid(PortableCollectionPFRecHitFractionSoALayout).name());
//
// using PortableCollectionEcalDigiSoALayout = PortableCollection<EcalDigiSoALayout<128, false>, Device>;
// DEFINE_EDM_PLUGIN(ngt::SerialiserFactory,
//                   ngt::Serialiser<PortableCollectionEcalDigiSoALayout>,
//                   typeid(PortableCollectionEcalDigiSoALayout).name());
//
// using PortableCollectionEcalUncalibratedRecHitSoALayout =
//     PortableCollection<EcalUncalibratedRecHitSoALayout<128, false>, Device>;
// DEFINE_EDM_PLUGIN(ngt::SerialiserFactory,
//                   ngt::Serialiser<PortableCollectionEcalUncalibratedRecHitSoALayout>,
//                   typeid(PortableCollectionEcalUncalibratedRecHitSoALayout).name());
