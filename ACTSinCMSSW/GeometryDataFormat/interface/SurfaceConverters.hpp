#ifndef SURFACE_CONVERTERS_HPP
#define SURFACE_CONVERTERS_HPP

// ACTS:
#include "Acts/Surfaces/PlaneSurface.hpp"
#include "Acts/Surfaces/RectangleBounds.hpp"
#include "Acts/Surfaces/TrapezoidBounds.hpp"
#include "Acts/Geometry/SurfaceArrayCreator.hpp"
// CMSSW:
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
// Custom:
#include "ACTSinCMSSW/GeometryDataFormat/interface/CMSDetectorElement.h"

class SurfaceConverters {

    public:
    SurfaceConverters(const TrackerGeometry* TrkGeo);

    std::shared_ptr<Acts::Surface> fromCMSSWtoACTS(const Surface& surf_cmssw);

    std::shared_ptr<Surface> fromACTStoCMSSW(const Acts::Surface& surf_acts);

    private:
    const TrackerGeometry* m_trackerGeometry;

};
 


#endif