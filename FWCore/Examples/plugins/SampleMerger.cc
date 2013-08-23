// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/transform.h"

#include "DataFormats/Examples/interface/SampleProduct.h"

//
// class declaration
//

class SampleMerger : public edm::stream::EDProducer<> {
  public:
    explicit SampleMerger(edm::ParameterSet const & config);
    ~SampleMerger() = default;

    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  private:
    virtual void produce(edm::Event & event, edm::EventSetup const & setup) override;

    const std::vector<edm::EDGetTokenT<example::SampleProductCollection>>   m_source;
};


SampleMerger::SampleMerger(const edm::ParameterSet& config) :
  m_source(
    // fill the input tokens
    edm::vector_transform(
      config.getParameter<std::vector<edm::InputTag>>("source"),
      [this](edm::InputTag const & tag) { return this->consumes<example::SampleProductCollection>(tag); }
    )
  )
{
  // register produced products
  produces<example::SampleProductCollection>();
}


void
SampleMerger::produce(edm::Event & event, edm::EventSetup const & setup)
{
  auto product = std::make_unique<example::SampleProductCollection>();
  edm::Handle<example::SampleProductCollection> handle;
  for (auto const & source : m_source) {
    event.getByToken(source, handle);
    for (auto const & element : *handle)
      product->push_back(element);
  }
  event.put(std::move(product));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
SampleMerger::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.add("source", std::vector<edm::InputTag>());
  descriptions.add("sampleMerger", desc);
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SampleMerger);
