#include "HLTrigger/Phase2HLTPFTaus/plugins/PATTauEvenOddEventMixer.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include <assert.h>

PATTauEvenOddEventMixer::PATTauEvenOddEventMixer(const edm::ParameterSet& cfg) 
 : moduleLabel_(cfg.getParameter<std::string>("@module_label"))
 , numEvenEvents_(0)
 , numEvenTaus_(0)
 , numOddEvents_(0)
 , numOddTaus_(0)
{
  srcEvenEvents_ = cfg.getParameter<edm::InputTag>("srcEvenEvents");
  tokenEvenEvents_ = consumes<pat::TauCollection>(srcEvenEvents_);
  srcOddEvents_ = cfg.getParameter<edm::InputTag>("srcOddEvents");
  tokenOddEvents_ = consumes<pat::TauCollection>(srcOddEvents_);

  produces<pat::TauCollection>();
}

PATTauEvenOddEventMixer::~PATTauEvenOddEventMixer()
{
  std::cout << "<PATTauEvenOddEventMixer (moduleLabel = '" << moduleLabel_ << "')>:" << std::endl;
  std::cout << " #even events processed = " << numEvenEvents_ << " (#taus = " << numEvenTaus_ << ")" << std::endl;
  std::cout << " #odd events processed = " << numOddEvents_ << " (#taus = " << numOddTaus_ << ")" << std::endl;
}

void PATTauEvenOddEventMixer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  std::unique_ptr<pat::TauCollection> pfTaus_output(new pat::TauCollection());

  edm::Handle<pat::TauCollection> pfTaus_input;
  if ( (evt.id().event() % 2) == 0 ) 
  {
    evt.getByToken(tokenEvenEvents_, pfTaus_input);
    ++numEvenEvents_;
    numEvenTaus_ += pfTaus_input->size();
  }
  else if ( (evt.id().event() % 2) == 1 ) 
  {
    evt.getByToken(tokenOddEvents_, pfTaus_input);
    ++numOddEvents_;
    numOddTaus_ += pfTaus_input->size();
  }
  else assert(0);
  pfTaus_output->insert(pfTaus_output->end(), pfTaus_input->begin(), pfTaus_input->end());

  evt.put(std::move(pfTaus_output));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATTauEvenOddEventMixer);
