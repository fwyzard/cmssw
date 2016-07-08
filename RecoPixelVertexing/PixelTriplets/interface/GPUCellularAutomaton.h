#ifndef GPUCELLULARAUTOMATON_H_
#define GPUCELLULARAUTOMATON_H_

#include <array>

#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
#include "RecoPixelVertexing/PixelTriplets/interface/GPUSimpleVector.h"

#include <cuda.h>



template<unsigned int theNumberOfLayers, unsigned int maxNumberOfQuadruplets>
class GPUCellularAutomaton {
public:

    GPUCellularAutomaton(TrackingRegion const & region, float thetaCut, float phiCut) :
      thePtMin{ region.ptMin() },
      theRegionOriginX{ region.origin().x() },
      theRegionOriginY{ region.origin().y() },
      theRegionOriginRadius{ region.originRBound() },
      theThetaCut{ thetaCut },
      thePhiCut{ phiCut }
    {
    }

    void run(std::array<const GPULayerDoublets *, theNumberOfLayers-1> const & doublets);

private:

    float thePtMin;
    float theRegionOriginX;
    float theRegionOriginY;
    float theRegionOriginRadius;
    float theThetaCut;
    float thePhiCut;
};

#endif
