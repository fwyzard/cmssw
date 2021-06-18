#ifndef HLTrigger_Phase2HLTPFTaus_EmptyCollectionProducer_h
#define HLTrigger_Phase2HLTPFTaus_EmptyCollectionProducer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

template <class T, class TCollection = std::vector<T>>
class EmptyCollectionProducer : public edm::global::EDProducer<>
{
 public:
  explicit EmptyCollectionProducer(const edm::ParameterSet& cfg)
  {
    produces<TCollection>();
  }
  ~EmptyCollectionProducer() override
  {}

  void produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& es) const 
  {
    std::unique_ptr<TCollection> objects(new TCollection());
    evt.put(std::move(objects));
  }
};

#endif
