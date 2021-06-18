#include "HLTrigger/Phase2HLTPFTaus/plugins/EmptyCollectionProducer.h"

#include "DataFormats/PatCandidates/interface/Electron.h"      // pat::Electron
#include "DataFormats/PatCandidates/interface/Muon.h"          // pat::Muon
#include "DataFormats/PatCandidates/interface/Tau.h"           // pat::Tau
#include "DataFormats/PatCandidates/interface/Jet.h"           // pat::Jet
#include "DataFormats/PatCandidates/interface/IsolatedTrack.h" // pat::IsolatedTrack

typedef EmptyCollectionProducer<pat::Electron>      EmptyPATElectronCollectionProducer;
typedef EmptyCollectionProducer<pat::Muon>          EmptyPATMuonCollectionProducer;
typedef EmptyCollectionProducer<pat::Tau>           EmptyPATTauCollectionProducer;
typedef EmptyCollectionProducer<pat::Jet>           EmptyPATJetCollectionProducer;
typedef EmptyCollectionProducer<pat::IsolatedTrack> EmptyPATIsolatedTrackCollectionProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(EmptyPATElectronCollectionProducer);
DEFINE_FWK_MODULE(EmptyPATMuonCollectionProducer);
DEFINE_FWK_MODULE(EmptyPATTauCollectionProducer);
DEFINE_FWK_MODULE(EmptyPATJetCollectionProducer);
DEFINE_FWK_MODULE(EmptyPATIsolatedTrackCollectionProducer);
