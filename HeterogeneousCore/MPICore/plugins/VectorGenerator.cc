// -*- C++ -*-
//
// Package:    HeterogeneousCore/MPICore
// Class:      VectorGenerator
//
/**\class VectorGenerator VectorGenerator.cc HeterogeneousCore/MPICore/plugins/VectorGenerator.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Fawaz M Sh Kh W Alazemi
//         Created:  Tue, 11 Jun 2024 12:52:13 GMT
//
//

// system include files
#include <memory>
#include <vector>
#include <random>
#include <ctime>
#include <cstdlib>
#include "TBufferFile.h"
#include "TClass.h"
#include "TString.h"

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
template <typename T>
void sendSerializedProduct(T& product) {
  static const TClass* type = TClass::GetClass<T>();
  TBufferFile buffer(TBuffer::kWrite);
  buffer.WriteClassBuffer(type, (&product));

  // Send the serialized buffer
  // MPI_Send(buffer.Buffer(), buffer.Length(), MPI_BYTE, destination, tag, comm);
}

class VectorGenerator : public edm::stream::EDProducer<> {
public:
  explicit VectorGenerator(const edm::ParameterSet&);
  ~VectorGenerator() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  //void beginRun(edm::Run const&, edm::EventSetup const&) override;
  //void endRun(edm::Run const&, edm::EventSetup const&) override;
  //void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
  edm::EDPutTokenT<std::vector<int>> vectorToken_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
VectorGenerator::VectorGenerator(const edm::ParameterSet& iConfig) : vectorToken_{produces()} {
  //register your products
  /* Examples
  produces<ExampleData2>();

  //if do put with a label
  produces<ExampleData2>("label");
 
  //if you want to put into the Run
  produces<ExampleData2,InRun>();
  */
  //now do what ever other initialization is needed
}

VectorGenerator::~VectorGenerator() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  //
  // please remove this method altogether if it would be left empty
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void VectorGenerator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
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
  // Function to generate a specified number of random integers between 5 and 20 and print them
  std::srand(std::time(0));
  int randomSize = std::rand() % 16 + 5;  //random size from 5 to 20

  std::vector<int> V;
  V.reserve(randomSize);

  for (int i = 0; i < randomSize; ++i) {
    int r = std::rand() % 10000;
    V.push_back(r);
  }

  iEvent.emplace(vectorToken_, V);
  sendSerializedProduct(V);
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void VectorGenerator::beginStream(edm::StreamID) {
  // please remove this method if not needed
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void VectorGenerator::endStream() {
  // please remove this method if not needed
}

// ------------ method called when starting to processes a run  ------------
/*
void
VectorGenerator::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
VectorGenerator::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
VectorGenerator::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
VectorGenerator::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void VectorGenerator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(VectorGenerator);
