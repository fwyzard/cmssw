#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Sources/interface/ProducerSourceBase.h"

/*
 * ProducerSourceBase inherits from IDGeneratorSourceBase<PuttableSourceBase>
 *
 * IDGeneratorSourceBase implements the logic to generate run, lumi, and event numbers, and event timestamps.
 * These will actually be overwritten by this source, but it's easier to do that than to write a new source base
 * type from scratch.
 *
 * PuttableSourceBase implements and provides a produce() method to let this source put the token into the event.
 */

class ConfigurableSource : public edm::ProducerSourceBase {
public:
  explicit ConfigurableSource(edm::ParameterSet const&, edm::InputSourceDescription const&);
  ~ConfigurableSource() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool setRunAndEventInfo(edm::EventID& id, edm::TimeValue_t& time, edm::EventAuxiliary::ExperimentType&) override;
  void produce(edm::Event&) override;

  std::vector<edm::EventID> events_;
  edm::EDPutTokenT<edmtest::IntProduct> token_;

  int counter_ = 0;
};

ConfigurableSource::ConfigurableSource(edm::ParameterSet const& config,
                                       edm::InputSourceDescription const& desc)
    :  // note that almost all configuration parameters passed to IDGeneratorSourceBase via ProducerSourceBase will
       // effectively be ignored, because this ConfigurableSource will explicitly set the run, lumi, and event
       // numbers, the timestamp, and the event type
      edm::ProducerSourceBase(config, desc, false),
      events_{config.getUntrackedParameter<std::vector<edm::EventID>>("events")},  // list of event ids to create
      token_{produces<edmtest::IntProduct>()}                                      // counter stored in each event
{
  // invert the order of ht events so they can efficiently be popped off the back of the vector
  std::reverse(events_.begin(), events_.end());
}

bool ConfigurableSource::setRunAndEventInfo(edm::EventID& event,
                                            edm::TimeValue_t& time,
                                            edm::EventAuxiliary::ExperimentType& type) {
  if (events_.empty()) {
    return false;
  }

  event = std::move(events_.back());
  events_.pop_back();
  return true;
}

void ConfigurableSource::produce(edm::Event& event) { event.put(std::make_unique<edmtest::IntProduct>(counter_++)); }

void ConfigurableSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Creates runs, lumis and events (containing no products) based on the provided list of events.");
  edm::ProducerSourceBase::fillDescription(desc);

  desc.addUntracked<std::vector<edm::EventID>>("events", {});
  descriptions.add("source", desc);
}

#include "FWCore/Framework/interface/InputSourceMacros.h"
DEFINE_FWK_INPUT_SOURCE(ConfigurableSource);
