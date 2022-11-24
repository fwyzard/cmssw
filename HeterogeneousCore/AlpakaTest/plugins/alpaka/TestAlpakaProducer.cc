#include "DataFormats/Portable/interface/Product.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "TestAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TestAlpakaProducer : public stream::EDProducer<> {
  public:
    TestAlpakaProducer(edm::ParameterSet const& config)
        : deviceToken_{produces()},
          deviceTokenMulti_{produces()},
          size_{config.getParameter<int32_t>("size")},
          size2_{config.getParameter<int32_t>("size2")} {}

    void produce(device::Event& event, device::EventSetup const&) override {
      // run the algorithm, potentially asynchronously
      portabletest::TestDeviceCollection deviceProduct{size_, event.queue()};
      portabletest::TestDeviceMultiCollection deviceMultiProduct{{{size_, size2_}}, event.queue()};
      algo_.fill(event.queue(), deviceProduct);
      algo_.fillMulti(event.queue(), deviceMultiProduct);

      // put the asynchronous product into the event without waiting
      event.emplace(deviceToken_, std::move(deviceProduct));
      event.emplace(deviceTokenMulti_, std::move(deviceMultiProduct));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<int32_t>("size");
      desc.add<int32_t>("size2");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::EDPutToken<portabletest::TestDeviceCollection> deviceToken_;
    const device::EDPutToken<portabletest::TestDeviceMultiCollection> deviceTokenMulti_;
    const int32_t size_;
    const int32_t size2_;

    // implementation of the algorithm
    TestAlgo algo_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestAlpakaProducer);
