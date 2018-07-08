/** \class EventTimestamp
 */

#include <iomanip>
#include <iostream>

#include <sys/time.h>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"

//
// class declaration
//

class EventTimestamp : public edm::global::EDAnalyzer<edm::RunCache<void>> {

 public:
  explicit EventTimestamp(const edm::ParameterSet&);
  ~EventTimestamp() override;
  void globalBeginRun(edm::Run const&, edm::EventSetup const&) const final;
  void globalEndRun(edm::Run const&, edm::EventSetup const&) const final;
  void analyze(edm::StreamID, edm::Event const& , edm::EventSetup const&) const final;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

 private:
  //edm::InputTag                          rawDataCollection_;
  //edm::EDGetTokenT<FEDRawDataCollection> rawDataToken_;
};

//
// constructors and destructor
//
EventTimestamp::EventTimestamp(const edm::ParameterSet& ps) // :
  //rawDataCollection_( ps.getParameter<edm::InputTag>("RawDataCollection") ),
  //rawDataToken_(      consumes<FEDRawDataCollection>(rawDataCollection_) )
{
}

EventTimestamp::~EventTimestamp() = default;

void
EventTimestamp::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  //desc.add<edm::InputTag>("RawDataCollection", edm::InputTag("rawDataCollector"));
  descriptions.add("eventTimestamp", desc);
}

//
// member functions
//

// ------------ method called to analyze the data  ------------
void
EventTimestamp::globalBeginRun(edm::Run const & run, edm::EventSetup const & setup) const
{
  long begin_s  = run.beginTime().unixTime();
  long begin_us = run.beginTime().microsecondOffset();
  edm::TimeOfDay begin_timestamp({begin_s, begin_us});
  long end_s    = run.endTime().unixTime();
  long end_us   = run.endTime().microsecondOffset();
  edm::TimeOfDay end_timestamp({end_s, end_us});
  std::cout << "run " << run.run() << " started at " << begin_timestamp << " and ended at " << end_timestamp "." << std::endl;
}

void
EventTimestamp::globalEndRun(edm::Run const & run, edm::EventSetup const & setup) const
{
  long begin_s  = run.beginTime().unixTime();
  long begin_us = run.beginTime().microsecondOffset();
  edm::TimeOfDay begin_timestamp({begin_s, begin_us});
  long end_s    = run.endTime().unixTime();
  long end_us   = run.endTime().microsecondOffset();
  edm::TimeOfDay end_timestamp({end_s, end_us});
  std::cout << "run " << run.run() << " started at " << begin_timestamp << " and ended at " << end_timestamp "." << std::endl;
}

void
EventTimestamp::analyze(edm::StreamID sid, edm::Event const & event, edm::EventSetup const & setup) const
{
  //edm::Handle<FEDRawDataCollection> rawDataHandle ;
  //event.getByToken(rawDataToken_, rawDataHandle );
  long event_s  = event.time().unixTime();
  long event_us = event.time().microsecondOffset();
  edm::TimeOfDay event_timestamp({event_s, event_us});

  double delta = (event.bunchCrossing() / 3564. + event.orbitNumber()) / 11245.5;
  long delta_s  = static_cast<long>(std::trunc(delta));
  long delta_us = static_cast<long>(std::nearbyint((delta - delta_s) * 1000000));
  event_s  -= delta_s;
  event_us -= delta_us;
  if (event_us < 0) {
    event_s  -= 1;
    event_us += 1000000;
  }
  edm::TimeOfDay run_timestamp({event_s, event_us});

  std::cout << "run " << event.run() << ", lumisection " << event.luminosityBlock() << ", event " << event.id().event() << ": orbit " << event.orbitNumber() << ", bx " << std::setw(4) << std::setfill(' ') << event.bunchCrossing() << ", time stamp: " << event_timestamp << std::endl;
  std::cout << "estimated run start at " << run_timestamp << std::endl;
}


// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EventTimestamp);
