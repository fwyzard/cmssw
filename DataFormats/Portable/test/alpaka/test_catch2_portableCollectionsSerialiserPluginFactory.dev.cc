#include <alpaka/alpaka.hpp>
#include <catch.hpp>

#include "DataFormats/Common/interface/TrivialCopyTraits.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "Eigen/src/Core/Matrix.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "TrivialSerialisation/Common/interface/alpaka/SerialiserFactory.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

/*
* This test demonstrates how TrivialSerialiser can be utilized to copy a PortableCollection of an arbitrary type "T" of which only its runtime type information is known. The portable collection is accessed via a WrapperBase pointer to Wrapper<T>.
*
* This test performs the following steps:
* - Create a PortableCollection and initialize it with some data.
* - Create an empty PortableCollection of the same size, into which the data will be copied.
* - Wrap both collections in `WrapperBase*` so they can can be handled without knowing their types. 
* - Use the TrivialSerialiser interface to copy the data from one into another.
* - Check that the data was successfully copied
*
*/

TEST_CASE("Test TrivialCopyTraits", "[TrivialCopyTraits]") {
  // initialize the edmplugin::PluginManager
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend");
  }

  SECTION("PortableCollection<portabletest::TestSoALayout<128, false>, Device>") {
    using PortableCollectionType = PortableCollection<portabletest::TestSoALayout<128, false>, Device>;
    using PortableHostCollectionType = PortableHostCollection<portabletest::TestSoALayout<128, false>>;
    const int size = 10;

    for (auto const& device : devices) {
      std::cout << "Running on " << alpaka::getName(device) << std::endl;
      Queue queue(device);

      // Create a PortableHostCollection, to be used as a reference for all checks
      PortableHostCollectionType refHostCollection(size, queue);

      // Initialize it with some data
      refHostCollection.view().r() = 3.14;
      for (int i = 0; i < size; i++) {
        refHostCollection.view()[i].x() = i * size + 1;
        refHostCollection.view()[i].y() = i * size + 2;
        refHostCollection.view()[i].z() = i * size + 3;
        refHostCollection.view()[i].id() = i;
        refHostCollection.view()[i].flags() = std::array<short, 4>{
            {static_cast<short>(i), static_cast<short>(i + 1), static_cast<short>(i + 2), static_cast<short>(i + 3)}};
        refHostCollection.view()[i].m = Eigen::Matrix<double, 3, 6>::Identity() * i;
      }

      auto checkPortableHostCollection = [&queue, &refHostCollection](PortableCollectionType& col) {
        // Since PortableCollection might be a device collection, copy first its data into an auxiliary PortableHostCollection
        PortableHostCollectionType auxPortableHostCollection(size, queue);
        alpaka::memcpy(queue, auxPortableHostCollection.buffer(), col.buffer());
        alpaka::wait(queue);

        printf("Comparing to the reference host collection\n");
        REQUIRE(auxPortableHostCollection.const_view().r() == refHostCollection.const_view().r());
        for (int i = 0; i < size; i++) {
          REQUIRE(auxPortableHostCollection.const_view()[i].x() == refHostCollection.const_view()[i].x());
          REQUIRE(auxPortableHostCollection.const_view()[i].y() == refHostCollection.const_view()[i].y());
          REQUIRE(auxPortableHostCollection.const_view()[i].z() == refHostCollection.const_view()[i].z());
          REQUIRE(auxPortableHostCollection.const_view()[i].id() == refHostCollection.const_view()[i].id());
          REQUIRE(auxPortableHostCollection.const_view()[i].flags() == refHostCollection.const_view()[i].flags());
          REQUIRE(auxPortableHostCollection.const_view()[i].m() == refHostCollection.const_view()[i].m());
        }
      };

      // Create the PortableCollection from which data will be cloned, and initialize it with the reference data
      PortableCollectionType sourcePortableCollection(size, queue);
      alpaka::memcpy(queue, sourcePortableCollection.buffer(), refHostCollection.buffer());

      // Check that sourcePortableCollection has been successfully initialized.
      checkPortableHostCollection(sourcePortableCollection);

      // Create an uninitialized clone.
      PortableCollectionType clonePortableCollection(edm::Uninitialized{});

      // Wrap both collections
      edm::Wrapper<PortableCollectionType> wrapper_original(
          std::make_unique<PortableCollectionType>(std::move(sourcePortableCollection)));
      edm::Wrapper<PortableCollectionType> wrapper_clone(
          std::make_unique<PortableCollectionType>(std::move(clonePortableCollection)));

      // Now cast each wrapper to edm::WrapperBase, hiding the underlying collection type
      edm::WrapperBase const* wb_original = static_cast<const edm::WrapperBase*>(&wrapper_original);
      edm::WrapperBase* wb_clone = static_cast<edm::WrapperBase*>(&wrapper_clone);

      // Check if there is a TrivialSerialiser plugin for type "PortableCollectionType"
      static_assert(edm::HasTrivialCopyTraits<PortableCollectionType>);

      // Get the plugin
      std::string typeName = typeid(PortableCollectionType).name();
      std::unique_ptr<ngt::SerialiserBase> serialiserSource{
          ngt::SerialiserFactoryPortable::get()->create(typeName)};

      // Initialize serialisers
      auto reader = serialiserSource->initialize(*wb_original);
      auto writer = serialiserSource->initialize(*wb_clone);


      // Initialize the clone with properties from the original. In practice, this will resize the clone portable collection to match the size of the original.
      writer->initialize(reader->parameters(), queue);

      // Get memory regions
      auto targets = writer->regions();
      auto sources = reader->regions();

      // Check that reader and writer have the same number of memory regions. In the case of a PortableCollection this should be equal to one.
      REQUIRE(sources.size() == targets.size());

      // copy each region from the source to the clone
      for (size_t i = 0; i < sources.size(); ++i) {
        REQUIRE(sources[i].data() != nullptr);
        REQUIRE(targets[i].data() != nullptr);
        REQUIRE(targets[i].size_bytes() == sources[i].size_bytes());

        auto src_view = alpaka::createView(device, sources[i].data(), sources[i].size_bytes());
        auto trg_view = alpaka::createView(device, targets[i].data(), targets[i].size_bytes());

        alpaka::memcpy(queue, trg_view, src_view);
      }
      alpaka::wait(queue);

      // Get the PortableCollection from the clone's wrapper and check that the copy succeeded.
      checkPortableHostCollection(wrapper_clone.bareProduct());
    }
  }
}
