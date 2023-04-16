// C++ headers
#include <algorithm>
#include <numeric>

// Alpaka lib
//#include <alpaka/alpaka.hpp>

// CMSSW headers
#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"

#include "PixelRecHitGPUKernel.h"
#include "pixelRecHits.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools;
  template <typename TrackerTraits>
  class setHitsLayerStart {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  uint32_t const* __restrict__ hitsModuleStart,
                                  pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits> const* cpeParams,
                                  uint32_t* hitsLayerStart) const {
      const int32_t i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];
      constexpr auto m = TrackerTraits::numberOfLayers;

      assert(0 == hitsModuleStart[0]);

      if (i <= (int32_t)m) {
        hitsLayerStart[i] = hitsModuleStart[cpeParams->layerGeometry().layerStart[i]];
#ifdef GPU_DEBUG
        int old = i == 0 ? 0 : hitsModuleStart[cpeParams->layerGeometry().layerStart[i - 1]];
        printf("LayerStart %d/%d at module %d: %d - %d\n",
               i,
               m,
               cpeParams->layerGeometry().layerStart[i],
               hitsLayerStart[i],
               hitsLayerStart[i] - old);
#endif
      }
    }
  };

  namespace pixelgpudetails {

    template <typename TrackerTraits>
    TrackingRecHitAlpakaDevice<TrackerTraits> PixelRecHitGPUKernel<TrackerTraits>::makeHitsAsync(
        SiPixelDigisDevice const& digis_d,
        SiPixelClustersDevice const& clusters_d,
        BeamSpotAlpaka const& bs_d,
        pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits> const* cpeParams,
        Queue queue) const {
      using namespace gpuPixelRecHits;
      auto nHits = clusters_d.nClusters();

      TrackingRecHitAlpakaDevice<TrackerTraits> hits_d(
          nHits, clusters_d.offsetBPIX2(), cpeParams, clusters_d->clusModuleStart(), queue);

      int activeModulesWithDigis = digis_d.nModules();
      // protect from empty events
      if (activeModulesWithDigis) {
        int threadsPerBlock = 128;
        int blocks = activeModulesWithDigis;
        const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

#ifdef GPU_DEBUG
        std::cout << "launching getHits kernel for " << blocks << " blocks" << std::endl;
#endif
        alpaka::exec<Acc1D>(queue,
                            workDiv1D,
                            getHits<TrackerTraits>{},
                            cpeParams,
                            bs_d.data(),
                            digis_d.view(),
                            digis_d.nDigis(),
                            clusters_d.view(),
                            hits_d.view());
        // getHits<TrackerTraits><<<blocks, threadsPerBlock, 0, stream>>>(
        //     cpeParams, bs_d.data(), digis_d.view(), digis_d.nDigis(), clusters_d.const_view(), hits_d.view());
#ifdef GPU_DEBUG
        alpaka::wait(queue);
#endif

        // assuming full warp of threads is better than a smaller number...
        if (nHits) {
          const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(1, 32);
          // setHitsLayerStart<TrackerTraits>
          //     <<<1, 32, 0, stream>>>(clusters_d->clusModuleStart(), cpeParams, hits_d.view().hitsLayerStart().data());
          alpaka::exec<Acc1D>(queue,
                              workDiv1D,
                              setHitsLayerStart<TrackerTraits>{},
                              clusters_d->clusModuleStart(),
                              cpeParams,
                              hits_d.view().hitsLayerStart().data());
          constexpr auto nLayers = TrackerTraits::numberOfLayers;
          cms::alpakatools::fillManyFromVector<Acc1D>(&(hits_d.view().phiBinner()),
                                                      nLayers,
                                                      hits_d.view().iphi(),
                                                      hits_d.view().hitsLayerStart().data(),
                                                      nHits,
                                                      (uint32_t)256,
                                                      queue);

#ifdef GPU_DEBUG
          alpaka::wait(queue);
#endif
        }
      }

#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "PixelRecHitGPUKernel -> DONE!" << std::endl;
#endif

      return hits_d;
    }

    template class PixelRecHitGPUKernel<pixelTopology::Phase1>;
    template class PixelRecHitGPUKernel<pixelTopology::Phase2>;

  }  // namespace pixelgpudetails
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
