#include <string>

#include <alpaka/alpaka.hpp>

#include "DataFormats/XyzId/interface/alpaka/XyzIdAlpakaDeviceCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/config.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/device.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/host.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/queue.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class XyzIdAlpakaProducer : public edm::stream::EDProducer<> {
  public:
    XyzIdAlpakaProducer(edm::ParameterSet const& config)
        : deviceToken_{produces<XyzIdAlpakaDeviceCollection>()}, size_{config.getParameter<int32_t>("size")} {}

    void produce(edm::Event& event, edm::EventSetup const&) override {
      XyzIdAlpakaDeviceCollection deviceProduct{size_, device};
      for (int32_t i = 0; i < size_; ++i) {
        // write values to the device buffer one at a time
        int32_t value = i;
        auto hostView = alpaka::createView(host, &value, Vec1D{1});
        auto deviceView = alpaka::createView(device, &deviceProduct->id(i), Vec1D{1});
        alpaka::memcpy(queue, deviceView, hostView);
      }
      event.emplace(deviceToken_, std::move(deviceProduct));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<int32_t>("size");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const edm::EDPutTokenT<XyzIdAlpakaDeviceCollection> deviceToken_;
    const int32_t size_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/macros.h"
DEFINE_FWK_ALPAKA_MODULE(XyzIdAlpakaProducer);
