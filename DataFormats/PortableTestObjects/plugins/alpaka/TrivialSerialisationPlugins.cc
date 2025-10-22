#include "DataFormats/Portable/interface/PortableHostObject.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "TrivialSerialisation/Common/interface/alpaka/SerialiserFactory.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"
#include "DataFormats/PortableTestObjects/interface/TestStruct.h"

#include "DataFormats/Common/interface/DeviceProduct.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableObject.h"
#include "DataFormats/Portable/interface/PortableDeviceObject.h"

#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using DeviceProductPortableObjectTestStruct = edm::DeviceProduct<PortableObject<portabletest::TestStruct, Device>>;
  DEFINE_TRIVIAL_SERIALISER_PLUGIN(DeviceProductPortableObjectTestStruct);

  using DeviceProductPortableCollectionTestSoALayout =
      edm::DeviceProduct<PortableCollection<portabletest::TestSoALayout<128, false>, Device>>;
  DEFINE_TRIVIAL_SERIALISER_PLUGIN(DeviceProductPortableCollectionTestSoALayout);

  using PortableCollectionTestSoALayout =
      PortableCollection<portabletest::TestSoALayout<128, false>, Device>;
  DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableCollectionTestSoALayout);

  using PortableMultiCollection2 = edm::DeviceProduct<
      PortableMultiCollection<Device, portabletest::TestSoALayout<128, false>, portabletest::TestSoALayout2<128, false>>>;
  DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableMultiCollection2);

  using PortableMultiCollection3 =
      edm::DeviceProduct<PortableMultiCollection<Device,
                                                 portabletest::TestSoALayout<128, false>,
                                                 portabletest::TestSoALayout2<128, false>,
                                                 portabletest::TestSoALayout3<128, false>>>;
  DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableMultiCollection3);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
