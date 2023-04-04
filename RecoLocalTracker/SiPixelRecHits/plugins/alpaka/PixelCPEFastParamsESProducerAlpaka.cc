#include <memory>
#include <string>
#include <alpaka/alpaka.hpp>
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFastAlpaka.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFastParams.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  template <typename TrackerTraits>
  class PixelCPEFastParamsESProducerAlpaka : public ESProducer {
  public:
    PixelCPEFastParamsESProducerAlpaka(edm::ParameterSet const& iConfig);
    std::optional<pixelCPEforDevice::PixelCPEFastParams<Device, TrackerTraits>> produce(device::EventSetup& es);
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    const device::ESGetToken<PixelCPEFastAlpaka<TrackerTraits>, TkPixelCPERecord> cpeToken_;
  };

  template <typename TrackerTraits>
  PixelCPEFastParamsESProducerAlpaka<TrackerTraits>::PixelCPEFastParamsESProducerAlpaka(edm::ParameterSet const& iConfig)
      : ESProducer(iConfig) {
    auto const& myname = iConfig.getParameter<std::string>("ComponentName");
  }

  template <typename TrackerTraits>
  std::optional<pixelCPEforDevice::PixelCPEFastParams<Device, TrackerTraits>>
  PixelCPEFastParamsESProducerAlpaka<TrackerTraits>::produce(device::EventSetup& es) {
    auto const& pcpe = es.getData(cpeToken_);
    auto pcpe_buf = cms::alpakatools::make_host_buffer<pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits>, Platform>();
    memcpy(pcpe_buf.data(), &pcpe.getCPEFastParams(), sizeof(pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits>));

    return pixelCPEforDevice::PixelCPEFastParams<Device, TrackerTraits>(pcpe_buf);
  }

  template <typename TrackerTraits>
  void PixelCPEFastParamsESProducerAlpaka<TrackerTraits>::fillDescriptions(
      edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    std::string name = "PixelCPEFast";
    name += TrackerTraits::nameModifier;
    desc.add<std::string>("ComponentName", name);

    descriptions.addWithDefaultLabel(desc);
  }

  using PixelCPEFastParamsESProducerAlpakaPhase1 = PixelCPEFastParamsESProducerAlpaka<pixelTopology::Phase1>;
  using PixelCPEFastParamsESProducerAlpakaPhase2 = PixelCPEFastParamsESProducerAlpaka<pixelTopology::Phase2>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(PixelCPEFastParamsESProducerAlpakaPhase1);
DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(PixelCPEFastParamsESProducerAlpakaPhase2);
