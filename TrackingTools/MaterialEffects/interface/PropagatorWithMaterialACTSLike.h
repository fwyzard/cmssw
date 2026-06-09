#ifndef _COMMONRECO_PropagatorWithMaterialACTSLIKE_H_
#define _COMMONRECO_PropagatorWithMaterialACTSLIKE_H_


#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"
#include "TrackPropagation/RungeKutta/interface/defaultRKPropagator.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/MaterialEffects/interface/MaterialEffectsUpdator.h"

//
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "ACTSinCMSSW/GeometryDataFormat/interface/CMSMagneticFieldProvider.hpp"
#include "ACTSinCMSSW/GeometryDataFormat/interface/CMSDetectorElement.h"
#include "ACTSinCMSSW/GeometryDataFormat/interface/SurfaceConverters.hpp"
#include "TrackingTools/MaterialEffects/interface/ConcretePropagator.hpp"
#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Propagator/MaterialInteractor.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Propagator/detail/SteppingLogger.hpp"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/MaterialEffects/interface/CombinedMaterialEffectsUpdator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "TrackingTools/GeomPropagators/interface/PropagationDirectionFromPath.h"
#include <string>

using DetElVect = std::vector<std::shared_ptr<Acts::CMSDetectorElement>>;
using MyConcretePropagator = ConcretePropagator<Acts::Propagator<Acts::EigenStepper<>, Acts::Navigator>>;
using TsosWP = std::pair<TrajectoryStateOnSurface, double>;
struct TrackingGeometryWithDetEls {
    DetElVect detElements;
    std::shared_ptr<Acts::TrackingGeometry> trackingGeometry;
};
TYPELOOKUP_DATA_REG(TrackingGeometryWithDetEls);

class MagneticField;
class PropagatorWithMaterialACTSLike final : public Propagator {
public:

  PropagatorWithMaterialACTSLike(PropagationDirection dir,
                         const float mass,
                         const MagneticField* mf = nullptr,
                         const float maxDPhi = 1.6,
                         bool useRungeKutta = false,
                         float ptMin = -1.,
                         bool useOldGeoPropLogic = true,
                         //
                         std::shared_ptr<TrackingGeometryWithDetEls> TrkandDetEls = nullptr,
                         const TrackerGeometry* trkGeo_cmssw = nullptr);

  ~PropagatorWithMaterialACTSLike() override;

  using Propagator::propagate;
  using Propagator::propagateWithPath;

private:
  
  TsosWP getTsosWP_fromACTS_params(Acts::Vector4 pos4, Acts::Vector3 dir, double qOverP,  Acts::BoundMatrix covMat_acts, const Surface& surfTarget) const;
  
  std::pair<TrajectoryStateOnSurface, double> propagateWithPath(const TrajectoryStateOnSurface& tsos,
                                                                const Plane& plane) const override;

  std::pair<TrajectoryStateOnSurface, double> propagateWithPath(const FreeTrajectoryState& fts,
                                                                const Plane& plane) const override;

  std::pair<TrajectoryStateOnSurface, double> propagateWithPath(const TrajectoryStateOnSurface& tsos,
                                                                const Cylinder& cylinder) const override;

  std::pair<TrajectoryStateOnSurface, double> propagateWithPath(const FreeTrajectoryState& fts,
                                                                const Cylinder& cylinder) const override;                                                             

public:


  /// Limit on change in azimuthal angle
  bool setMaxDirectionChange(float phiMax) override { return theGeometricalPropagator->setMaxDirectionChange(phiMax); }

  enum MaterialLocation { atSource, atDestination, fromDirection };
  /** Choice of location for including material effects:
   *  fromDirection is equivalent to atSource for propagation alongMomentum
   *  and to atDestination for propagation oppositeToMomentum.
   *  Inclusion of material effects at the source (either explicitely or
   *  implicitely) is not possible if propagating with anyDirection and
   *  will effectively disable material effects when propagating from
   *  a FreeTrajectoryState.
   */
  void setMaterialLocation(const MaterialLocation location) { theMaterialLocation = location; }


  
  /// Access to the geometrical propagator
  const Propagator& geometricalPropagator() const { return *theGeometricalPropagator; }
  /// Access to the MaterialEffectsUpdator
  const MaterialEffectsUpdator& materialEffectsUpdator() const { return *theMEUpdator; }


  const MagneticField* magneticField() const override { return field; }

  PropagatorWithMaterialACTSLike* clone() const override { return new PropagatorWithMaterialACTSLike(*this); }

private:
  /// Inclusion of material at the source?
  bool materialAtSource() const dso_internal;

private:
  // Geometrical propagator

  defaultRKPropagator::Product rkProduct;
  DeepCopyPointerByClone<Propagator> theGeometricalPropagator;

  // Material effects
  DeepCopyPointerByClone<MaterialEffectsUpdator> theMEUpdator;
  // Use material at source?
  MaterialLocation theMaterialLocation;
  const MagneticField* field;
  bool useRungeKutta_;

  //
  std::shared_ptr<TrackingGeometryWithDetEls> trkGeo_and_DetEls_;
  const TrackerGeometry* trkGeo_cmssw_;
  DeepCopyPointerByClone<MyConcretePropagator> ConcProp;
};

#endif
