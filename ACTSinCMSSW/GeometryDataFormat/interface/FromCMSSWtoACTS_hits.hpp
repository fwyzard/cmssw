#ifndef FROM_CMSSW_TO_ACTS_HITS
#define FROM_CMSSW_TO_ACTS_HITS

#pragma once

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/TrackParametrization.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/EventData/SourceLink.hpp"
#include <memory>
#include <iostream>

#include "ACTSinCMSSW/GeometryDataFormat/interface/CMSDetectorSourceLink.hpp"
#include "ACTSinCMSSW/GeometryDataFormat/interface/CMSDetectorElement.h"

using DetElVect = std::vector<std::shared_ptr<Acts::CMSDetectorElement>>;

class FromCMSSWtoACTS_hits {
  public:

    struct HitsInfo {
        // Info on Hit Type
        /// NOTE: PixelBarrel = 1; PixelEndcap = 2; TIB = 3; TID = 4; TOB = 5; TEC = 6;
        std::vector<int> hit_type;
        // Info on the local position of the hits
        std::vector<double> loc_x;
        std::vector<double> loc_y;
        std::vector<double> loc_x_err;
        std::vector<double> loc_y_err;
        // Module ids of the hits
        std::vector<uint32_t> mod_ids;
    };

    explicit FromCMSSWtoACTS_hits(HitsInfo hits_info, DetElVect detEl_vec);

    ~FromCMSSWtoACTS_hits() = default;

    std::vector<Acts::SourceLink> convert();

  private:
  std::unordered_map<uint32_t, std::shared_ptr<Acts::CMSDetectorElement> > geoIdToDetElMap;
  HitsInfo hits_info_;
  DetElVect detEl_vec_;


};


inline FromCMSSWtoACTS_hits::FromCMSSWtoACTS_hits(HitsInfo hits_info, DetElVect detEl_vec)
    : hits_info_(hits_info),
      detEl_vec_(detEl_vec){
    for(const auto& detEl : detEl_vec_){
        geoIdToDetElMap[(*detEl).detID()] = detEl;
    }
}

inline std::vector<Acts::SourceLink> FromCMSSWtoACTS_hits::convert() {

    std::vector<Acts::SourceLink> sl_vector;

    // Loop on the hits collection
    for(std::size_t i = 0; i < hits_info_.loc_x.size(); i++) {
        CMSDetectorSourceLink DetLink;

        // ==== Hit type ====
        if(hits_info_.hit_type[i] == 1 || hits_info_.hit_type[i] == 2){
            // std::cout << "[DEBUG] PIXEL -> local x: " << hits_info_.loc_x[i] << "+-"<< "; local y: " << hits_info_.loc_y[i] << std::endl;
            DetLink.hitType = HitType::Hit2D; // 2D
            // ==== Position ====
            DetLink.lPos[0] = hits_info_.loc_x[i];
            DetLink.lPos[1] = hits_info_.loc_y[i];
            // ==== Covariance ====
            DetLink.lCov << pow(hits_info_.loc_x_err[i],2), 0,  
                                                        0, pow(hits_info_.loc_y_err[i],2);
            // DetLink.lCov << hits_info_.loc_x_err[i], 0,  
            //                                             0, hits_info_.loc_y_err[i];
            
        } else {
            // std::cout << "[DEBUG] STRIP -> local x: " << hits_info_.loc_x[i] << "; local y: " << hits_info_.loc_y[i] << std::endl;
            DetLink.hitType = HitType::Hit1D; // 1D
            // ==== Position ====
            DetLink.lPos[0] = hits_info_.loc_x[i];
            DetLink.lPos[1] = hits_info_.loc_y[i];

            // ==== Covariance ====
            DetLink.lCov << pow(hits_info_.loc_x_err[i],2), 0, 
                                                        0,  0;
            // DetLink.lCov << hits_info_.loc_x_err[i], 0,
            //                                       0,  0;
        }

        // ==== Surface ====
        auto it = geoIdToDetElMap.find(hits_info_.mod_ids[i]);
        if(it != geoIdToDetElMap.end()){ 
            auto detEl = it->second;
            DetLink.surf = (*detEl).surface().getSharedPtr();
        } else {
            std::cout << ">>>> ERROR: Surface Not found for this hit!" << std::endl;
        }

        Acts::SourceLink sourceLink(DetLink);
        sl_vector.push_back(sourceLink);
    }

    return sl_vector;
}


#endif  // FROM_CMSSW_TO_ACTS_HITS



