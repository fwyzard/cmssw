#ifndef RECORDS_ACTSTRACKERGEOMETRYRECORD_H
#define RECORDS_ACTSTRACKERGEOMETRYRECORD_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurfaceDeformationRcd.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"
#include "Geometry/Records/interface/PTrackerAdditionalParametersPerDetRcd.h"
#include "FWCore/Utilities/interface/mplVector.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

class ACTSTrackerGeometryRecord
    : public edm::eventsetup::DependentRecordImplementation<ACTSTrackerGeometryRecord,
                                                            edm::mpl::Vector<TrackerDigiGeometryRecord,
                                                                             TrackerTopologyRcd,
                                                                             TrackerAlignmentRcd> > {
};

#endif /* RECORDS_ACTSTRACKERGEOMETRYRECORD_H */
