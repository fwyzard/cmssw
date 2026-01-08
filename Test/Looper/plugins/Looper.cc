#include <iostream>

#include "FWCore/Framework/interface/EDLooper.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ModuleChanger.h"
#include "FWCore/Framework/interface/ScheduleInfo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class Looper : public edm::EDLooper {
public:
  explicit Looper(const edm::ParameterSet & config) :
    repeat_{config.getParameter<uint32_t>("repeat")},
    module_{config.getParameter<std::string>("module")},
    parameter_{config.getParameter<std::string>("parameter")}
  {}
  ~Looper() override = default;

  void beginOfJob() override {
    std::cerr << "Looper::beginOfJob()\n";

    // access the original configuration of "module"
    const edm::ParameterSet* old_config = scheduleInfo()->parametersForModule(module_);
    assert(old_config);

    // check that "module" has the parameter "parameter"
    (void) old_config->getParameter<int32_t>(parameter_);

    // make a copy of the original configuration
    config_ = *old_config;
  }

  void endOfJob() override {
    std::cerr << "Looper::endOfJob()\n";
  }

  void startingNewLoop(unsigned int counter) override {
    std::cerr << "Looper::startingNewLoop(" << counter << ")\n";
  }

  edm::EDLooper::Status duringLoop(const edm::Event & event, const edm::EventSetup &) override {
    std::cerr << "Looper::duringLoop(" << event.id() << ", setup)\n";
    return kContinue;
  }

  edm::EDLooper::Status endOfLoop(const edm::EventSetup &, unsigned int counter) override {
    std::cerr << "Looper::endOfLoop(setup, " << counter << ")\n";
    if (repeat_ != 0 and counter >= repeat_) {
      return kStop;
    }

    config_.addParameter<int32_t>(parameter_, config_.getParameter<int32_t>(parameter_) + 1);
    bool success = moduleChanger()->changeModule(module_, config_);
    assert(success);

    return kContinue;
  }

private:
  uint32_t repeat_;
  std::string module_;
  std::string parameter_;

  edm::ParameterSet config_;
};

#include "FWCore/Framework/interface/LooperFactory.h"
DEFINE_FWK_LOOPER(Looper);
