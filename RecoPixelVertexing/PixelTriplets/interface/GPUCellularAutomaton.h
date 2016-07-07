#ifndef GPUCELLULARAUTOMATON_H_
#define GPUCELLULARAUTOMATON_H_


#include <array>
#include "GPUCACell.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"


template<unsigned int theNumberOfLayers>
class GPUCellularAutomaton {
public:

    GPUCellularAutomaton(std::vector<const HitDoublets*> doublets, const TrackingRegion& region, const float phiCut, const float thetaCut) {



    }

    void create_and_connect_cells();


    void evolve();


    void find_ntuplets(std::vector<GPUCACell<theNumberOfLayers>::CAntuplet>&, const unsigned int);



private:

    GPU_HitDoublets* gpuDoublets;
    RecHitsSortedInPhi_gpu* hits;
    GPUCACell<numberOfLayers>** theCells;
    RecHitsSortedInPhi_gpu* hitsOnLayers;
    GPUArena<numberOfLayers-1, 4, GPUCACell<numberOfLayers>* >* isOuterHitOfCell;
    GPUArena<numberOfLayers,4,GPUCACell<numberOfLayers>* >* theInnerNeighbors;
    GPUSimpleVector<maxNumberOfQuadruplets, CAntuplet>* foundNtuplets;


};


#endif
