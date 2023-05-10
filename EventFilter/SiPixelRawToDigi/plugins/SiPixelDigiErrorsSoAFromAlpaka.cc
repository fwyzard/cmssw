#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsHost.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelErrorsSoA.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/SimpleVector.h"

class SiPixelDigiErrorsSoAFromAlpaka : public edm::stream::EDProducer<> {
public:
  explicit SiPixelDigiErrorsSoAFromAlpaka(const edm::ParameterSet& iConfig);
  ~SiPixelDigiErrorsSoAFromAlpaka() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<SiPixelDigiErrorsHost> digiErrorGetToken_;
  edm::EDPutTokenT<SiPixelErrorsSoA> digiErrorPutToken_;

  std::optional<cms::alpakatools::host_buffer<SiPixelErrorCompact[]>> data_;
  cms::alpakatools::SimpleVector<SiPixelErrorCompact> error_ =
      cms::alpakatools::make_SimpleVector<SiPixelErrorCompact>(0, nullptr);
  const SiPixelFormatterErrors* formatterErrors_ = nullptr;
};

SiPixelDigiErrorsSoAFromAlpaka::SiPixelDigiErrorsSoAFromAlpaka(const edm::ParameterSet& iConfig)
    : digiErrorGetToken_(consumes<SiPixelDigiErrorsHost>(iConfig.getParameter<edm::InputTag>("src"))),
      digiErrorPutToken_(produces<SiPixelErrorsSoA>()) {}

void SiPixelDigiErrorsSoAFromAlpaka::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("siPixelClustersAlpaka"));
  descriptions.addWithDefaultLabel(desc);
}

void SiPixelDigiErrorsSoAFromAlpaka::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // The following line copies the data from the pinned host memory to
  // regular host memory. In principle that feels unnecessary (why not
  // just use the pinned host memory?). There are a few arguments for
  // doing it though
  // - Now can release the pinned host memory back to the (caching) allocator
  //   * if we'd like to keep the pinned memory, we'd need to also
  //     keep the CUDA stream around as long as that, or allow pinned
  //     host memory to be allocated without a CUDA stream
  // - What if a CPU algorithm would produce the same SoA? We can't
  //   use cudaMallocHost without a GPU...

  const auto& cpuDigiErrors = iEvent.get(digiErrorGetToken_);

  if (cpuDigiErrors.nErrorWords() == 0)
    return;

  error_ = *cpuDigiErrors.error();
  data_ = std::move(cpuDigiErrors.error_data());
  formatterErrors_ = &(cpuDigiErrors.formatterErrors());

  iEvent.emplace(digiErrorPutToken_, error_.size(), error_.data(), formatterErrors_);
  error_ = cms::alpakatools::make_SimpleVector<SiPixelErrorCompact>(0, nullptr);
  data_.reset();
  formatterErrors_ = nullptr;
}

// define as framework plugin
DEFINE_FWK_MODULE(SiPixelDigiErrorsSoAFromAlpaka);