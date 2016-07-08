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

/*
template<unsigned int theNumberOfLayers, unsigned int maxNumberOfQuadruplets>
void
GPUCellularAutomaton<theNumberOfLayers, maxNumberOfQuadruplets>::run(std::array<const GPULayerDoublets *, theNumberOfLayers-1> const & doublets)
{
  int numberOfChunksIn1stArena = 0;
  std::array<int, theNumberOfLayers-1> numberOfKeysIn1stArena;
  for (size_t i = 0; i < theNumberOfLayers-1; ++i) {
    numberOfKeysIn1stArena[i] = doublets[i]->layers[1].size;
    numberOfChunksIn1stArena += doublets[i]->size;
  }
  //GPUArena<theNumberOfLayers-1, 4, GPUCACell<theNumberOfLayers>* > isOuterHitOfCell(numberOfChunksIn1stArena, numberOfKeysIn1stArena);
  GPUArena<theNumberOfLayers-1, 4, GPUCACell<theNumberOfLayers>> isOuterHitOfCell(numberOfChunksIn1stArena, numberOfKeysIn1stArena);

  int numberOfChunksIn2ndArena = 0;
  std::array<int, theNumberOfLayers-2> numberOfKeysIn2ndArena;
  for (size_t i = 1; i < theNumberOfLayers-1; ++i) {
    numberOfKeysIn2ndArena[i] = doublets[i]->size;
    numberOfChunksIn2ndArena += doublets[i-1]->size;
  }
  //GPUArena<theNumberOfLayers-2, 4, GPUCACell<theNumberOfLayers>* > theInnerNeighbors(numberOfChunksIn2ndArena, numberOfKeysIn2ndArena);
  GPUArena<theNumberOfLayers-2, 4, GPUCACell<theNumberOfLayers>> theInnerNeighbors(numberOfChunksIn2ndArena, numberOfKeysIn2ndArena);

  GPUCACell<theNumberOfLayers>* theCells[theNumberOfLayers-1];
  for (unsigned int i = 0; i< theNumberOfLayers-1; ++i)
    cudaMalloc(& theCells[i], doublets[i]->size * sizeof(GPUCACell<theNumberOfLayers>));

  GPUSimpleVector<maxNumberOfQuadruplets, CAntuplet>* foundNtuplets;
  cudaMalloc(& foundNtuplets, sizeof(GPUSimpleVector<maxNumberOfQuadruplets, CAntuplet>));
}
*/
#endif
