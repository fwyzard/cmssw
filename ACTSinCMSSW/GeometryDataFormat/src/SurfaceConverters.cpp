#include "ACTSinCMSSW/GeometryDataFormat/interface/SurfaceConverters.hpp"


SurfaceConverters::SurfaceConverters(const TrackerGeometry* TrkGeo)
    : m_trackerGeometry(TrkGeo) {}

std::shared_ptr<Acts::Surface> SurfaceConverters::fromCMSSWtoACTS(const Surface& surf_cmssw) {
    
    // ===== Define the transformation (i.e. Rotation and Translation) of the CMSSW surface =====
    const auto& pos = surf_cmssw.position();
    const auto& rot = surf_cmssw.rotation();
    Acts::Transform3 t = Acts::Transform3::Identity();
    Acts::RotationMatrix3 R;
    R << rot.xx(), rot.yx(), rot.zx(), rot.xy(), rot.yy(), rot.zy(), rot.xz(), rot.yz(), rot.zz();
    t.prerotate(R);
    t.pretranslate(Acts::Vector3(pos.x()*10, pos.y()*10, pos.z()*10)); // from cm to mm

    // ===== Define the ACTS surface considering two tipes of bounds (i.e. Rectangle and Trapezoid) =====
    std::shared_ptr<Acts::Surface> acts_surf = nullptr;
    auto bounds = dynamic_cast<const RectangularPlaneBounds*>(&surf_cmssw.bounds());
    if (bounds){
      const std::size_t kValues = Acts::RectangleBounds::BoundValues::eSize;
      std::array<double, kValues> bValues{};
      std::vector<double> bVector = {-bounds->width()  / 2 * 10,  // cm → mm
                                     -bounds->length() / 2 * 10,
                                      bounds->width()  / 2 * 10,
                                      bounds->length() / 2 * 10};

      std::copy_n(bVector.begin(), kValues, bValues.begin());
      acts_surf = Acts::Surface::makeShared<Acts::PlaneSurface>(t, std::move(std::make_shared<const Acts::RectangleBounds>(bValues)));
    }
    else{
      auto trap = dynamic_cast<const TrapezoidalPlaneBounds*>(&surf_cmssw.bounds());
      auto params = trap->parameters();

      const std::size_t kValues = Acts::TrapezoidBounds::BoundValues::eSize;
      std::array<double, kValues> bValues{};
      std::vector<double> bVector = {params[3] * 10, // cm to mm
                                     params[3] * 10, 
                                     (params[0] + params[1]) / 2 * 10, 
                                     0.0};

      std::copy_n(bVector.begin(), kValues, bValues.begin());
      acts_surf = Acts::Surface::makeShared<Acts::PlaneSurface>(t, std::move(std::make_shared<const Acts::TrapezoidBounds>(bValues)));
    }

    return acts_surf;
}

std::shared_ptr<Surface> SurfaceConverters::fromACTStoCMSSW(const Acts::Surface& surf_acts) {
    
    // Get the detector elemeent associated to the surface to obtain the detId
    auto detEl = dynamic_cast<const Acts::CMSDetectorElement*>(surf_acts.associatedDetectorElement());
    auto detID = detEl->detID();

    // Get the GeoDet unit from the detId
    const GeomDet* geomDet = m_trackerGeometry->idToDet(detID);    

    return std::make_shared<Plane>(geomDet->specificSurface()); 
}
