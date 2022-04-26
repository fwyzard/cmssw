// The "Transcriber" makes sense only across different memory spaces
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) or defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#include <string>

#include <alpaka/alpaka.hpp>

#include "DataFormats/XyzId/interface/XyzIdAlpakaHostCollection.h"
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

  class XyzIdAlpakaTranscriber : public edm::stream::EDProducer<> {
  public:
    XyzIdAlpakaTranscriber(edm::ParameterSet const& config)
        : deviceToken_{consumes<XyzIdAlpakaDeviceCollection>(config.getParameter<edm::InputTag>("source"))},
          hostToken_{produces<XyzIdAlpakaHostCollection>()} {}

    void produce(edm::Event& event, edm::EventSetup const&) override {
      XyzIdAlpakaDeviceCollection const& deviceProduct = event.get(deviceToken_);

      XyzIdAlpakaHostCollection hostProduct{deviceProduct->size(), host, device};
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
    const edm::EDGetTokenT<XyzIdAlpakaDeviceCollection> deviceToken_;
    const edm::EDPutTokenT<XyzIdAlpakaHostCollection> hostToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/macros.h"
DEFINE_FWK_ALPAKA_MODULE(XyzIdAlpakaTranscriber);

#endif  // defined(ALPAKA_ACC_GPU_CUDA_ENABLED) or defined(ALPAKA_ACC_GPU_HIP_ENABLED)
