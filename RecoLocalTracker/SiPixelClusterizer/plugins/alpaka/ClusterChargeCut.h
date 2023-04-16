#ifndef RecoLocalTracker_SiPixelClusterizer_alpaka_ClusterChargeCut_h
#define RecoLocalTracker_SiPixelClusterizer_alpaka_ClusterChargeCut_h

#include <cstdint>
#include <cstdio>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/SiPixelClusterThresholds.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersLayout.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisLayout.h"

// namespace ALPAKA_ACCELERATOR_NAMESPACE {
namespace pixelClustering {

  template <typename TrackerTraits>
  struct clusterChargeCut {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc& acc,
        SiPixelDigisLayoutSoAView digi_view,
        SiPixelClustersLayoutSoAView clus_view,
        SiPixelClusterThresholds
            clusterThresholds,  // charge cut on cluster in electrons (for layer 1 and for other layers)
        const uint32_t numElements) const {
      constexpr int startBPIX2 = TrackerTraits::layerStart[1];
      [[maybe_unused]] constexpr int nMaxModules = TrackerTraits::numberOfModules;

      const uint32_t blockIdx(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
      if (blockIdx >= clus_view[0].moduleStart())
        return;

      auto firstPixel = clus_view[1 + blockIdx].moduleStart();
      auto thisModuleId = digi_view[firstPixel].moduleId();

      ALPAKA_ASSERT_OFFLOAD(nMaxModules < maxNumModules);
      ALPAKA_ASSERT_OFFLOAD(startBPIX2 < nMaxModules);

      auto nclus = clus_view[thisModuleId].clusInModule();
      if (nclus == 0)
        return;

      const uint32_t threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      if (threadIdxLocal == 0 && nclus > maxNumClustersPerModules)
        printf("Warning too many clusters in module %d in block %d: %d > %d\n",
               thisModuleId,
               blockIdx,
               nclus,
               maxNumClustersPerModules);

      // Stride = block size.
      const uint32_t blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);

      // Get thread / CPU element indices in block.
      const auto& [firstElementIdxNoStride, endElementIdxNoStride] =
          cms::alpakatools::element_index_range_in_block(acc, firstPixel);

      if (nclus > maxNumClustersPerModules) {
        uint32_t firstElementIdx = firstElementIdxNoStride;
        uint32_t endElementIdx = endElementIdxNoStride;
        // remove excess  FIXME find a way to cut charge first....
        for (uint32_t i = firstElementIdx; i < numElements; ++i) {
          if (not cms::alpakatools::next_valid_element_index_strided(
                  i, firstElementIdx, endElementIdx, blockDimension, numElements))
            break;
          if (digi_view[i].moduleId() == invalidModuleId)
            continue;  // not valid
          if (digi_view[i].moduleId() != thisModuleId)
            break;  // end of module
          if (digi_view[i].clus() >= maxNumClustersPerModules) {
            digi_view[i].moduleId() = invalidModuleId;
            digi_view[i].clus() = invalidModuleId;
          }
        }
        nclus = maxNumClustersPerModules;
      }

#ifdef GPU_DEBUG
      if (thisModuleId % 100 == 1)
        if (threadIdxLocal == 0)
          printf("start clusterizer for module %d in block %d\n", thisModuleId, blockIdx);
#endif

      auto& charge = alpaka::declareSharedVar<int32_t[maxNumClustersPerModules], __COUNTER__>(acc);
      auto& ok = alpaka::declareSharedVar<uint8_t[maxNumClustersPerModules], __COUNTER__>(acc);
      auto& newclusId = alpaka::declareSharedVar<uint16_t[maxNumClustersPerModules], __COUNTER__>(acc);

      ALPAKA_ASSERT_OFFLOAD(nclus <= maxNumClustersPerModules);
      cms::alpakatools::for_each_element_in_block_strided(acc, nclus, [&](uint32_t i) { charge[i] = 0; });
      alpaka::syncBlockThreads(acc);

      uint32_t firstElementIdx = firstElementIdxNoStride;
      uint32_t endElementIdx = endElementIdxNoStride;
      for (uint32_t i = firstElementIdx; i < numElements; ++i) {
        if (not cms::alpakatools::next_valid_element_index_strided(
                i, firstElementIdx, endElementIdx, blockDimension, numElements))
          break;
        if (digi_view[i].moduleId() == invalidModuleId)
          continue;  // not valid
        if (digi_view[i].moduleId() != thisModuleId)
          break;  // end of module
        alpaka::atomicAdd(
            acc, &charge[digi_view[i].clus()], static_cast<int32_t>(digi_view[i].adc()), alpaka::hierarchy::Threads{});
      }
      alpaka::syncBlockThreads(acc);

      auto chargeCut = clusterThresholds.getThresholdForLayerOnCondition(thisModuleId < startBPIX2);
      cms::alpakatools::for_each_element_in_block_strided(
          acc, nclus, [&](uint32_t i) { newclusId[i] = ok[i] = charge[i] > chargeCut ? 1 : 0; });
      alpaka::syncBlockThreads(acc);

      // renumber
      auto& ws = alpaka::declareSharedVar<uint16_t[32], __COUNTER__>(acc);
      cms::alpakatools::blockPrefixScan(acc, newclusId, nclus, ws);

      ALPAKA_ASSERT_OFFLOAD(nclus >= newclusId[nclus - 1]);

      if (nclus == newclusId[nclus - 1])
        return;

      clus_view[thisModuleId].clusInModule() = newclusId[nclus - 1];
      alpaka::syncBlockThreads(acc);

      // mark bad cluster again
      cms::alpakatools::for_each_element_in_block_strided(acc, nclus, [&](uint32_t i) {
        if (0 == ok[i])
          newclusId[i] = invalidModuleId + 1;
      });
      alpaka::syncBlockThreads(acc);

      // reassign id
      firstElementIdx = firstElementIdxNoStride;
      endElementIdx = endElementIdxNoStride;
      for (uint32_t i = firstElementIdx; i < numElements; ++i) {
        if (not cms::alpakatools::next_valid_element_index_strided(
                i, firstElementIdx, endElementIdx, blockDimension, numElements))
          break;
        if (digi_view[i].moduleId() == invalidModuleId)
          continue;  // not valid
        if (digi_view[i].moduleId() != thisModuleId)
          break;  // end of module
        digi_view[i].clus() = newclusId[digi_view[i].clus()] - 1;
        if (digi_view[i].clus() == invalidModuleId)
          digi_view[i].moduleId() = invalidModuleId;
      }

      //done
    }
  };

}  // namespace pixelClustering

#endif  //