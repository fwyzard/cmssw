#ifndef TrackingTools_ESProducers_PropagatorWithMaterialACTSLikeESProducer_h
#define TrackingTools_ESProducers_PropagatorWithMaterialACTSLikeESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecordForACTS.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "ACTSinCMSSW/GeometryDataFormat/interface/CMSDetectorElement.h"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Geometry/Records/interface/ACTSTrackerGeometryRecord.h"

#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterialACTSLike.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESConsumesCollector.h"

#include <memory>

class PropagatorWithMaterialACTSLikeESProducer : public edm::ESProducer {
public:
  explicit PropagatorWithMaterialACTSLikeESProducer(const edm::ParameterSet &p);

  std::unique_ptr<Propagator> produce(const TrackingComponentsRecordForACTS& iRecord);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:

  edm::ESGetToken<TrackingGeometryWithDetEls, ACTSTrackerGeometryRecord> ACTStrkGeomInfoToken_;  
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;

  double mass_;
  double maxDPhi_;
  double ptMin_;
  PropagationDirection dir_;
  bool useRK_;
  bool useOldAnalPropLogic_;
};

#endif
