#include "DataFormats/Common/interface/DeviceProduct.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceObject.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/SerialisationCore/interface/alpaka/SerialiserFactory.h"

DEFINE_PORTABLE_TRIVIAL_SERIALISER_PLUGIN(
    edm::DeviceProduct<ALPAKA_ACCELERATOR_NAMESPACE::portabletest::TestDeviceCollection>);
DEFINE_PORTABLE_TRIVIAL_SERIALISER_PLUGIN(
    edm::DeviceProduct<ALPAKA_ACCELERATOR_NAMESPACE::portabletest::TestDeviceMultiCollection2>);
DEFINE_PORTABLE_TRIVIAL_SERIALISER_PLUGIN(
    edm::DeviceProduct<ALPAKA_ACCELERATOR_NAMESPACE::portabletest::TestDeviceMultiCollection3>);
DEFINE_PORTABLE_TRIVIAL_SERIALISER_PLUGIN(
    edm::DeviceProduct<ALPAKA_ACCELERATOR_NAMESPACE::portabletest::TestDeviceObject>);
