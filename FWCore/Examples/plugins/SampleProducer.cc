// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/Examples/interface/SampleProduct.h"

//
// class declaration
//

class SampleProducer : public edm::stream::EDProducer<> {
  public:
    explicit SampleProducer(edm::ParameterSet const & config);
    ~SampleProducer() = default;

    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  private:
    virtual void produce(edm::Event & event, edm::EventSetup const & setup) override;

    const std::vector<std::string>  m_data;
};

SampleProducer::SampleProducer(const edm::ParameterSet& config) :
  m_data( config.getParameter<std::vector<std::string>>("data") )
{
  // register produced products
  produces<example::SampleProductCollection>();
}


void
SampleProducer::produce(edm::Event & event, edm::EventSetup const & setup)
{
  auto product = std::make_unique<example::SampleProductCollection>();
  for (auto const & element : m_data)
    product->push_back(example::SampleProduct(element));
  event.put(std::move(product));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
SampleProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.add("data", std::vector<std::string>());
  descriptions.add("sampleProducer", desc);
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SampleProducer);
