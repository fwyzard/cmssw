// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/Examples/interface/SampleProduct.h"

//
// class declaration
//

class SampleAnalyzer : public edm::stream::EDAnalyzer<> {
  public:
    explicit SampleAnalyzer(edm::ParameterSet const & config);
    ~SampleAnalyzer() = default;

    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  private:
    virtual void analyze(edm::Event const & event, edm::EventSetup const & setup) override;

    const edm::EDGetTokenT<example::SampleProductCollection>    m_source;
};

SampleAnalyzer::SampleAnalyzer(const edm::ParameterSet& config) :
  m_source( consumes<example::SampleProductCollection>(config.getParameter<edm::InputTag>("source")) )
{
}


void
SampleAnalyzer::analyze(edm::Event const & event, edm::EventSetup const & setup)
{
  edm::Handle<example::SampleProductCollection> handle;
  event.getByToken(m_source, handle);
  for (auto const & element : *handle)
    std::cout << element.data() << std::endl;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
SampleAnalyzer::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.add("source", edm::InputTag());
  descriptions.add("sampleAnalyzer", desc);
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SampleAnalyzer);
