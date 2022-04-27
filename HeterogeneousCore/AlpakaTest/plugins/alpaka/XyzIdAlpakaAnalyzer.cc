// The "Analyzer" makes sense only in the cpu memory space
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)

#include <cassert>
#include <string>
#include <iostream>

#include "DataFormats/XyzId/interface/XyzIdAlpakaHostCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class XyzIdAlpakaAnalyzer : public edm::stream::EDAnalyzer<> {
  public:
    XyzIdAlpakaAnalyzer(edm::ParameterSet const& config)
        : source_{config.getParameter<edm::InputTag>("source")}, token_{consumes<XyzIdAlpakaHostCollection>(source_)} {}

    void analyze(edm::Event const& event, edm::EventSetup const&) override {
      XyzIdAlpakaHostCollection const& product = event.get(token_);

      for (int32_t i = 0; i < product->size(); ++i) {
        //std::cout << source_ << "[" << i << "] = " << product->id(i) << std ::endl;
        assert(product->id(i) == i);
      }
      std::cout << "XyzIdAlpakaAnalyzer:\n"
                << source_.encode() << ".size() = " << product->size() << '\n'
                << std ::endl;
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("source");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const edm::InputTag source_;
    const edm::EDGetTokenT<XyzIdAlpakaHostCollection> token_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/macros.h"
DEFINE_FWK_ALPAKA_MODULE(XyzIdAlpakaAnalyzer);

#endif  // defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
