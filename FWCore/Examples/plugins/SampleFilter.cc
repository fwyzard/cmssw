// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/Examples/interface/SampleProduct.h"

//
// class declaration
//

class SampleFilter : public edm::stream::EDFilter<> {
  public:
    explicit SampleFilter(edm::ParameterSet const & config);
    ~SampleFilter() = default;

    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  private:
    virtual bool filter(edm::Event & event, edm::EventSetup const & setup) override;

    // ----------member data ---------------------------
    const edm::EDGetTokenT<example::SampleProductCollection>    m_source;
    const std::string                                           m_pattern;
};

SampleFilter::SampleFilter(const edm::ParameterSet& config) :
  m_source( consumes<example::SampleProductCollection>(config.getParameter<edm::InputTag>("source")) ),
  m_pattern( config.getParameter<std::string>("pattern") )
{
}


bool
SampleFilter::filter(edm::Event & event, edm::EventSetup const & setup)
{
  bool result = false;

  edm::Handle<example::SampleProductCollection> handle;
  event.getByToken(m_source, handle);
  for (auto const & element : *handle)
    if (m_pattern == element.data()) {
      result = true;
      break;
    }

  return result;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
SampleFilter::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.add("source",  edm::InputTag());
  desc.add("pattern", std::string());
  descriptions.add("sampleFilter", desc);
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SampleFilter);
