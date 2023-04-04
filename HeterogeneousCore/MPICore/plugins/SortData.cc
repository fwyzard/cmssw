// -*- C++ -*-
//
// Package:    HeterogeneousCore/SortData
// Class:      SortData
//
/**\class SortData SortData.cc HeterogeneousCore/MPICore/plugins/SortData.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Fawaz M Sh Kh W Alazemi
//         Created:  Thu, 17 Nov 2022 09:50:39 GMT
//
//

// system include files

#include <memory>
#include <sstream>
#include <vector>
#include <algorithm>

#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

//
// class declaration
//
//
//

class SortData : public edm::stream::EDProducer<> {
public:
  explicit SortData(const edm::ParameterSet&);
  ~SortData() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  //void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  //void endStream() override;

  //void beginRun(edm::Run const&, edm::EventSetup const&) override;
  //void endRun(edm::Run const&, edm::EventSetup const&) override;
  //void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<std::vector<int>> inData_;
  edm::EDPutTokenT<std::vector<int>> outData_;
  edm::StreamID sid_ = edm::StreamID::invalidStreamID();
};

//
// constants, enums and typedefs
//
/******************
 * MPI Begin
 ******************/

/*********** MPI END ***********/

//
// static data member definitions
//

//
// constructors and destructor
//
SortData::SortData(const edm::ParameterSet& iConfig)
    : inData_{consumes(iConfig.getParameter<edm::InputTag>("incomingData"))}, outData_{produces()} {
  //register your products
  /* Examples
  produces<ExampleData2>();

  //if do put with a label
  produces<ExampleData2>("label");
 
  //if you want to put into the Run
  produces<ExampleData2,InRun>();
  */
  //now do what ever other initialization is needed
  edm::LogAbsolute("MPI") << "SortData::SortData() is up.";
}

SortData::~SortData() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  //
  // please remove this method altogether if it would be left empty
  edm::LogAbsolute("MPI") << "SortData::~SortData()";
}

// ------------ method called to produce the data  ------------
void SortData::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;
  /* This is an event example
  //Read 'ExampleData' from the Event
  ExampleData const& in = iEvent.get(inToken_);
  
  //Use the ExampleData to create an ExampleData2 which 
  // is put into the Event
  iEvent.put(std::make_unique<ExampleData2>(in));
  */

  /* this is an EventSetup example
  //Read SetupData from the SetupRecord in the EventSetup
  SetupData& setup = iSetup.getData(setupToken_);
  */
  std::vector<int> data = iEvent.get(inData_);
  std::sort(data.begin(), data.end());

  std::stringstream ss;
  for (int i : data) {
    ss << i << " ";
  }

  edm::LogAbsolute("MPI") << "SortData::produce (sid_ = " << sid_.value() << ", Event = " << iEvent.id().event()
                          << ") Sorted Data = " << ss.str();
  iEvent.emplace(outData_, data);
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void SortData::beginStream(edm::StreamID stream) {
  // please remove this method if not needed
  sid_ = stream;
  edm::LogAbsolute("MPI") << "SortData::beginStream (Stream = " << sid_.value() << ").";
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
/*
void SortData::endStream() {
  // please remove this method if not needed
}
*/
// ------------ method called when starting to processes a run  ------------
/*
void
SortData::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
SortData::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
SortData::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
SortData::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SortData::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SortData);
