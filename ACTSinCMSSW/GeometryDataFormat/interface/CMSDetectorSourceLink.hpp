#ifndef CMS_DETECTOR_SOURCE_LINK_HPP
#define CMS_DETECTOR_SOURCE_LINK_HPP

#pragma once

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/TrackParametrization.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/EventData/SourceLink.hpp"
#include <memory>
#include <iostream>

using CovarianceMatrix = Eigen::Matrix<double, 2, 2>;
enum class HitType { Hit1D, Hit2D };

struct CMSDetectorSourceLink {

    HitType hitType;  //1D or 2D hit

    Acts::Vector2 lPos;          // local position           
    CovarianceMatrix lCov;       // local covariance           

    std::shared_ptr<const Acts::Surface> surf;        // Surface where the hit is recorded

    std::shared_ptr<const Acts::Surface> GetSurface() const { return surf;}
};

inline const Acts::Surface* CMSSurfaceAccessor(const Acts::SourceLink& sl) {
    auto& sl_type = sl.get<CMSDetectorSourceLink>();
    std::shared_ptr<const Acts::Surface> surf = sl_type.GetSurface();
    return surf.get(); 
}


#endif  // CMS_DETECTOR_SOURCE_LINK_HPP