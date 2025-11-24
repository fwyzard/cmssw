#include "DataFormats/PortableTestObjects/interface/TestHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/TestHostObject.h"
#include "HeterogeneousCore/SerialisationCore/interface/SerialiserFactory.h"

DEFINE_TRIVIAL_SERIALISER_PLUGIN(portabletest::TestHostCollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(portabletest::TestHostMultiCollection2);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(portabletest::TestHostMultiCollection3);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(portabletest::TestHostObject);
