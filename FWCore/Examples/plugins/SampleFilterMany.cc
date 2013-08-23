// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/transform.h"

#include "DataFormats/Examples/interface/SampleProduct.h"

//
// class declaration
//

class SampleFilterMany : public edm::stream::EDFilter<> {
  public:
    explicit SampleFilterMany(edm::ParameterSet const & config);
    ~SampleFilterMany() = default;

    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  private:
    virtual bool filter(edm::Event & event, edm::EventSetup const & setup) override;

    // ----------member data ---------------------------
    const std::vector<edm::EDGetTokenT<example::SampleProductCollection>>   m_source;
    const std::string                                                       m_pattern;
};

SampleFilterMany::SampleFilterMany(const edm::ParameterSet& config) :
  m_source(
    edm::vector_transform(
      config.getParameter<std::vector<edm::InputTag>>("source"),
      [this](edm::InputTag const & tag) { return this->mayConsume<example::SampleProductCollection>(tag); }
    )
  ),
  m_pattern( config.getParameter<std::string>("pattern") )
{
}


bool
SampleFilterMany::filter(edm::Event & event, edm::EventSetup const & setup)
{
  bool result = not m_source.empty();

  edm::Handle<example::SampleProductCollection> handle;
  for (auto const & source : m_source) {
    bool intermediate = false;
    event.getByToken(source, handle);
    for (auto const & element : *handle)
      if (m_pattern == element.data()) {
        intermediate = true;
        break;
      }
    result &= intermediate;
    if (not result) {
      // no need to read to other inputs
      break;
    }
  }

  return result;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
SampleFilterMany::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.add("source",  std::vector<edm::InputTag>());
  desc.add("pattern", std::string());
  descriptions.add("sampleFilterMany", desc);
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SampleFilterMany);
