#ifndef CMS_KF_CALIBRATOR_HPP
#define CMS_KF_CALIBRATOR_HPP

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/TrackParametrization.hpp"
#include "Acts/EventData/SourceLink.hpp"
#include "ACTSinCMSSW/GeometryDataFormat/interface/CMSDetectorSourceLink.hpp"
#include <iostream>
#include "Acts/Utilities/CalibrationContext.hpp"

template <typename traj_t>
void CMSKFCalibrator(const Acts::GeometryContext& gctx,
                   const Acts::CalibrationContext& cctx,
                   const Acts::SourceLink& sl,
                   typename traj_t::TrackStateProxy state) {

    const auto& myLink = sl.get<CMSDetectorSourceLink>();

    const auto& surf = myLink.GetSurface();

    if (myLink.hitType == HitType::Hit2D) {

        state.allocateCalibrated(2);
        state.template calibrated<2>() = myLink.lPos;
        state.template calibratedCovariance<2>() << myLink.lCov;
        state.setProjectorSubspaceIndices(std::array<Acts::BoundIndices, 2>{Acts::eBoundLoc0, Acts::eBoundLoc1});


        // std::cout << "[CAL] Calibrating 2D hit: pos = " << myLink.lPos << ", cov = " << myLink.lCov << std::endl;

    } else if (myLink.hitType == HitType::Hit1D){

        // vettore 1D
        Eigen::Matrix<double, 1, 1> pos1d;
        pos1d(0) = myLink.lPos[0];

        state.allocateCalibrated(1);
        state.template calibrated<1>() = pos1d;
        state.template calibratedCovariance<1>() << myLink.lCov(0,0);
        state.setProjectorSubspaceIndices(std::array<Acts::BoundIndices, 1>{Acts::eBoundLoc0});


        // std::cout << "[CAL] Calibrating 1D hit: pos = " << pos1d << ", cov = " << myLink.lCov << std::endl;
    } else {
        std::cout << "ERROR: Unrecognized hit type" << std::endl;
    }
}

#endif // CMS_KF_CALIBRATOR_HPP

