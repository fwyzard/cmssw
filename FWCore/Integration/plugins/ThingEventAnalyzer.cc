// CMSSW include files
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

// user include files

namespace edmtest {
  struct Empty {};
  class ThingEventAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    ThingEventAnalyzer(edm::ParameterSet const&);

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const final;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    edm::EDGetTokenT<ThingCollection> token_;
  };

  ThingEventAnalyzer::ThingEventAnalyzer(edm::ParameterSet const& config)
      : token_(consumes(config.getUntrackedParameter<edm::InputTag>("input"))) {}

  void ThingEventAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked("input", edm::InputTag{"thing", ""})->setComment("Collection to get from event");
    descriptions.add("thingEventAnalyzer", desc);
  }

  void ThingEventAnalyzer::analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const&) const {
    auto const& things = event.get(token_);
    {
      edm::LogAbsolute out("ThingEventAnalyzer");
      out << "found collection:";
      for (auto const& t : things) {
        out << '\t' << t.a;
      }
    }
  }

}  // namespace edmtest

#include "FWCore/Framework/interface/MakerMacros.h"
using edmtest::ThingEventAnalyzer;
DEFINE_FWK_MODULE(ThingEventAnalyzer);
