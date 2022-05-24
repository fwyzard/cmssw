// The "Transcriber" makes sense only across different memory spaces
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) or defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#include <optional>
#include <string>

#include <alpaka/alpaka.hpp>

#include "DataFormats/XyzId/interface/XyzIdHostCollection.h"
#include "DataFormats/XyzId/interface/alpaka/XyzIdDeviceCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaServices/interface/alpaka/AlpakaService.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class XyzIdAlpakaTranscriber : public edm::stream::EDProducer<> {
  public:
    XyzIdAlpakaTranscriber(edm::ParameterSet const& config)
        : deviceToken_{consumes<XyzIdDeviceCollection>(config.getParameter<edm::InputTag>("source"))},
          hostToken_{produces<XyzIdHostCollection>()} {}

    void beginStream(edm::StreamID sid) override {
      // choose a device based on the EDM stream number
      edm::Service<ALPAKA_TYPE_ALIAS(AlpakaService)> service;
      if (not service->enabled()) {
        throw cms::Exception("Configuration") << ALPAKA_TYPE_ALIAS_NAME(AlpakaService) << " is disabled.";
      }
      auto& devices = service->devices();
      unsigned int index = sid.value() % devices.size();
      device = devices[index];
    }

    void produce(edm::Event& event, edm::EventSetup const&) override {
      // create a queue to submit async work
      Queue queue{*device};
      XyzIdDeviceCollection const& deviceProduct = event.get(deviceToken_);

      XyzIdHostCollection hostProduct{deviceProduct->soaMetadata().size(), host, *device};
      alpaka::memcpy(queue, hostProduct.buffer(), deviceProduct.buffer());

      // wait for any async work to complete
      alpaka::wait(queue);

      event.emplace(hostToken_, std::move(hostProduct));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("source");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const edm::EDGetTokenT<XyzIdDeviceCollection> deviceToken_;
    const edm::EDPutTokenT<XyzIdHostCollection> hostToken_;

    // device associated to the EDM stream
    std::optional<Device> device;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(XyzIdAlpakaTranscriber);

#endif  // defined(ALPAKA_ACC_GPU_CUDA_ENABLED) or defined(ALPAKA_ACC_GPU_HIP_ENABLED)
