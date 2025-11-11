#include <cstring>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DataFormats/Common/interface/TrivialCopyTraits.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "TrivialSerialisation/Common/interface/SerialiserFactory.h"

/*
* This test demonstrates how TrivialSerialiser can be utilized to copy simple types TODO: fill this
*/

TEST_CASE("Test TrivialCopyTraits", "[TrivialCopyTraits]") {
  // initialize the edmplugin::PluginManager
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  SECTION("std::vector<float>") {
    const int size = 10;

    // Create a vector
    std::vector<float> vec(size);

    // Initialize the vector with some data
    for (int i = 0; i < size; i++) {
      vec[i] = static_cast<float>(i);
    }

    // Create an uninitialized clone.
    std::vector<float> vec_clone;

    // Wrap both vectors
    edm::Wrapper<std::vector<float>> wrapper_original(std::make_unique<std::vector<float>>(std::move(vec)));
    edm::Wrapper<std::vector<float>> wrapper_clone(std::make_unique<std::vector<float>>(std::move(vec_clone)));

    // Now cast each wrapper to edm::WrapperBase, hiding the underlying collection type
    edm::WrapperBase const* wb_original = static_cast<const edm::WrapperBase*>(&wrapper_original);
    edm::WrapperBase* wb_clone = static_cast<edm::WrapperBase*>(&wrapper_clone);

    // Check if there is a TrivialSerialiser plugin for type "std::vector<float>"
    static_assert(edm::HasTrivialCopyTraits<std::vector<float>>);

    // Get the plugin
    std::string typeName = typeid(std::vector<float>).name();
    // std::unique_ptr<ngt::SerialiserBase> serialiserSource{ngt::SerialiserFactory::get()->create(typeName)};

    std::unique_ptr<ngt::SerialiserBase> serialiser{ngt::SerialiserFactory::get()->tryToCreate(typeName)};

    if (serialiser) {
      edm::LogInfo("TrivialSerialisationTest") << "A serialiser plugin has been found for type " << typeName << ".";
    } else {
      edm::LogInfo("TrivialSerialisationTest") << "No serialiser plugin found for type " << typeName << ".";
    }

    REQUIRE(serialiser);

    // Initialize serialisers
    auto reader = serialiser->initialize(*wb_original);
    auto writer = serialiser->initialize(*wb_clone);

    // Initialize the clone with properties from the original.
    writer->initialize(reader->parameters());

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
      std::memcpy(targets[i].data(), sources[i].data(), sources[i].size_bytes());
    }

    // Check that the copy succeeded.
    auto const& vec_clone_ref = wrapper_clone.bareProduct();
    for (int i = 0; i < size; i++) {
      REQUIRE(vec_clone_ref[i] == static_cast<float>(i));
    }
  }
}
