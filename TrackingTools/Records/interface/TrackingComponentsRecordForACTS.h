#ifndef RecoTracker_Record_TrackingComponentsRecordForACTS_h
#define RecoTracker_Record_TrackingComponentsRecordForACTS_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
//#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/ACTSTrackerGeometryRecord.h"

#include "FWCore/Utilities/interface/mplVector.h"

class TrackingComponentsRecordForACTS 
    : public edm::eventsetup::DependentRecordImplementation<TrackingComponentsRecordForACTS,
                                        edm::mpl::Vector<IdealMagneticFieldRecord, 
                                                        GlobalTrackingGeometryRecord,
                                                        TrackerDigiGeometryRecord,
                                                        ACTSTrackerGeometryRecord> > {};

#endif