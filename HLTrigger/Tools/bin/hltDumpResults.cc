/* hltDumpResults: compare TriggerResults event by event */

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>

#include <TFile.h>
#include <TChain.h>

#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/ChainEvent.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/ChainEvent.h"
#include "HLTrigger/HLTcore/interface/HLTConfigData.h"


bool check_file(std::string const & file) {
  std::unique_ptr<TFile> f(TFile::Open(file.c_str()));
  return (f and not f->IsZombie());
}

bool check_files(std::vector<std::string> const & files) {
  bool flag = true;
  for (auto const & file: files) {
    if (not check_file(file)) {
      flag = false;
      std::cerr << "hltDumpResults: error: file " << file << " does not exist, or is not a ROOT file." << std::endl;
    }
  }
  return flag;
}


std::string getProcessNameFromBranch(std::string const & branch) {
  std::vector<boost::iterator_range<std::string::const_iterator>> tokens;
  boost::split(tokens, branch, boost::is_any_of("_."), boost::token_compress_off);
  return boost::copy_range<std::string>(tokens[3]);
}


std::unique_ptr<HLTConfigData> getHLTConfigData(fwlite::EventBase const & event, std::string process) {
  auto const & history = event.processHistory();
  if (process.empty()) {
    // determine the process name from the most recent "TriggerResults" object
    auto const & branch  = event.getBranchNameFor(edm::Wrapper<edm::TriggerResults>::typeInfo(), "TriggerResults", "", process.c_str());
    process = getProcessNameFromBranch(branch);
  }

  edm::ProcessConfiguration config;
  if (not history.getConfigurationForProcess(process, config)) {
    std::cerr << "error: the process " << process << " is not in the Process History" << std::endl;
    exit(1);
  }
  const edm::ParameterSet* pset = edm::pset::Registry::instance()->getMapped(config.parameterSetID());
  if (pset == nullptr) {
    std::cerr << "error: the configuration for the process " << process << " is not available in the Provenance" << std::endl;
    exit(1);
  }
  return std::make_unique<HLTConfigData>(pset);
}


int main(int argc, char ** argv) {
  std::vector<std::string> files;
  files.reserve(argc-1);
  for (int i = 1; i < argc; ++i)
    files.emplace_back(argv[i]);

  if (not check_files(files))
    exit(1);
  auto events = std::make_shared<fwlite::ChainEvent>(files);

  // initialise the HLT configuration
  std::unique_ptr<HLTConfigData> config;
  const std::string process = "HLT";

  const unsigned int num_events = events->size();
  const unsigned int denominator = std::max(1, static_cast<int>(num_events/100));
  unsigned int counter = 0;
  bool new_run = true;

  // loop over the events
  for (events->toBegin(); not events->atEnd(); ++(*events)) {
    // print progress every 1%
    if (counter % denominator == 0) {
      std::cerr << "Processed events: " << counter << " out of " << num_events << " (" << counter/denominator << "%)\r";
    }
    ++counter;

    // event id
    edm::EventID const& id = events->id();

    // read the trigger results
    fwlite::Handle<edm::TriggerResults> handle;
    edm::TriggerResults const * results = nullptr;
    handle.getByLabel<fwlite::Event>(* events->event(), "TriggerResults", "", process.c_str());
    if (handle.isValid())
      results = handle.product();
    else {
      std::cerr << "run " << id.run() << ", lumi " << id.luminosityBlock() << ", event " << id.event() << ": TriggerResults not found, skipping." << std::endl;
      continue;
    }

    // initialise the trigger configuration
    if (new_run) {
      new_run = false;
      events->fillParameterSetRegistry();
      config = getHLTConfigData(* events->event(), process);

      // print the trigger names
      for (auto const& name: config->triggerNames()) {
        std::cout << name << '\t';
      }
      std::cout << std::endl;
    }

    // print the trigger results
    for (unsigned int index = 0; index < config->size(); ++index) {
      std::cout << (results->state(index) == edm::hlt::Pass ? 1 : 0) << '\t';
    }
    std::cout << std::endl;
  }

  std::cerr << "Processed events: " << num_events << " out of " << num_events << " (100%)" << std::endl;
}

