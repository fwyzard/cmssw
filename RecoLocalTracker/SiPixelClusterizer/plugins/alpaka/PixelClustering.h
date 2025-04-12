#ifndef RecoLocalTracker_SiPixelClusterizer_alpaka_PixelClustering_h
#define RecoLocalTracker_SiPixelClusterizer_alpaka_PixelClustering_h

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"
#include "FWCore/Utilities/interface/DeviceGlobal.h"
#include "FWCore/Utilities/interface/HostDeviceConstant.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/SimpleVector.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/warpsize.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelImageSoA.h"

//#define GPU_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE::pixelClustering {

#ifdef GPU_DEBUG
  DEVICE_GLOBAL uint32_t gMaxHit = 0;
#endif

  namespace pixelStatus {
    // Phase-1 pixel modules
    constexpr uint32_t pixelSizeX = pixelTopology::Phase1::numRowsInModule;
    constexpr uint32_t pixelSizeY = pixelTopology::Phase1::numColsInModule;

    // Use 0x00, 0x01, 0x03 so each can be OR'ed on top of the previous ones
    enum Status : uint32_t { kEmpty = 0x00, kFound = 0x01, kDuplicate = 0x03 };

    constexpr uint32_t bits = 2;
    constexpr uint32_t mask = (0x01 << bits) - 1;
    constexpr uint32_t valuesPerWord = sizeof(uint32_t) * 8 / bits;
    constexpr uint32_t size = pixelSizeX * pixelSizeY / valuesPerWord;

    ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr uint32_t getIndex(uint16_t x, uint16_t y) {
      return (pixelSizeX * y + x) / valuesPerWord;
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr uint32_t getShift(uint16_t x, uint16_t y) {
      return (x % valuesPerWord) * 2;
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr Status getStatus(uint32_t const* __restrict__ status,
                                                              uint16_t x,
                                                              uint16_t y) {
      uint32_t index = getIndex(x, y);
      uint32_t shift = getShift(x, y);
      return Status{(status[index] >> shift) & mask};
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr bool isDuplicate(uint32_t const* __restrict__ status,
                                                              uint16_t x,
                                                              uint16_t y) {
      return getStatus(status, x, y) == kDuplicate;
    }

    /* FIXME
       * In the more general case (e.g. a multithreaded CPU backend) there is a potential race condition
       * between the read of status[index] at line NNN and the atomicCas at line NNN.
       * We should investigate:
       *   - if `status` should be read through a `volatile` pointer (CUDA/ROCm)
       *   - if `status` should be read with an atomic load (CPU)
       */
    ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr void promote(Acc1D const& acc,
                                                          uint32_t* __restrict__ status,
                                                          const uint16_t x,
                                                          const uint16_t y) {
      uint32_t index = getIndex(x, y);
      uint32_t shift = getShift(x, y);
      uint32_t old_word = status[index];
      uint32_t expected = old_word;
      do {
        expected = old_word;
        Status old_status{(old_word >> shift) & mask};
        if (kDuplicate == old_status) {
          // nothing to do
          return;
        }
        Status new_status = (kEmpty == old_status) ? kFound : kDuplicate;
        uint32_t new_word = old_word | (static_cast<uint32_t>(new_status) << shift);
        old_word = alpaka::atomicCas(acc, &status[index], expected, new_word, alpaka::hierarchy::Blocks{});
      } while (expected != old_word);
    }

  }  // namespace pixelStatus

  template <typename TrackerTraits>
  struct CountModules {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SiPixelDigisSoAView digi_view,
                                  SiPixelClustersSoAView clus_view,
                                  const unsigned int numElements) const {
      // Make sure the atomicInc below does not overflow
      static_assert(TrackerTraits::numberOfModules < ::pixelClustering::maxNumModules);

#ifdef GPU_DEBUG
      if (cms::alpakatools::once_per_grid(acc)) {
        printf("Starting to count modules to set module starts:");
      }
#endif
      for (int32_t i : cms::alpakatools::uniform_elements(acc, numElements)) {
        digi_view[i].clus() = i;
        if (::pixelClustering::invalidModuleId == digi_view[i].moduleId())
          continue;

        int32_t j = i - 1;
        while (j >= 0 and digi_view[j].moduleId() == ::pixelClustering::invalidModuleId)
          --j;
        if (j < 0 or digi_view[j].moduleId() != digi_view[i].moduleId()) {
          // Found a module boundary: count the number of modules in  clus_view[0].moduleStart()
          auto loc = alpaka::atomicInc(acc,
                                       &clus_view[0].moduleStart(),
                                       static_cast<uint32_t>(::pixelClustering::maxNumModules),
                                       alpaka::hierarchy::Blocks{});
          ALPAKA_ASSERT_ACC(loc < TrackerTraits::numberOfModules);
#ifdef GPU_DEBUG
          printf("> New module (no. %d) found at digi %d \n", loc, i);
#endif
          clus_view[loc + 1].moduleStart() = i;
        }
      }
    }
  };

  template <typename TrackerTraits>
  struct FindClus {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SiPixelDigisSoAView digi_view,
                                  SiPixelClustersSoAView clus_view,
                                  SiPixelImageSoAView images,
                                  const unsigned int numElements) const {
      static_assert(TrackerTraits::numberOfModules < ::pixelClustering::maxNumModules);

      // value used to mark empty pixels in the 2d image representation
      constexpr uint32_t kEmptyPixel = std::numeric_limits<uint16_t>::max();

      auto& lastPixel = alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);

      const uint32_t lastModule = clus_view[0].moduleStart();
      for (uint32_t module : cms::alpakatools::independent_groups(acc, lastModule)) {
        auto firstPixel = clus_view[1 + module].moduleStart();
        uint32_t thisModuleId = digi_view[firstPixel].moduleId();
        ALPAKA_ASSERT_ACC(thisModuleId < TrackerTraits::numberOfModules);

#ifdef GPU_DEBUG
        if (thisModuleId % 100 == 1)
          if (cms::alpakatools::once_per_block(acc))
            printf("start clusterizer for module %4d in block %4d\n",
                   thisModuleId,
                   alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
#endif

        // find the index of the first pixel not belonging to this module (or invalid)
        lastPixel = numElements;
        alpaka::syncBlockThreads(acc);

        // initialise the 2d image
        for (uint32_t i : cms::alpakatools::independent_group_elements(
                 acc, TrackerTraits::numRowsInModule * TrackerTraits::numColsInModule)) {
          uint32_t pos = module * TrackerTraits::numRowsInModule * TrackerTraits::numColsInModule + i;
          images[pos].id() = kEmptyPixel;
        }

        // skip threads not associated to an existing pixel
        for (uint32_t i : cms::alpakatools::independent_group_elements(acc, firstPixel, numElements)) {
          auto id = digi_view[i].moduleId();
          // skip invalid pixels
          if (id == ::pixelClustering::invalidModuleId)
            continue;
          // find the first pixel in a different module
          if (id != thisModuleId) {
            alpaka::atomicMin(acc, &lastPixel, i, alpaka::hierarchy::Threads{});
            break;
          }
        }

        // ensure that all threads agree on the value of lastPixel
        alpaka::syncBlockThreads(acc);
        ALPAKA_ASSERT_ACC((lastPixel == numElements) or
                          ((lastPixel < numElements) and (digi_view[lastPixel].moduleId() != thisModuleId)));

        // ensure that the total number of active pixels in the module can fit in 16 bit (- 1 for kEmptyPixel)
        ALPAKA_ASSERT_ACC((lastPixel - firstPixel) < kEmptyPixel);

        constexpr bool isPhase2 = std::is_base_of<pixelTopology::Phase2, TrackerTraits>::value;

        // fill the 2d image
        for (uint32_t i : cms::alpakatools::independent_group_elements(acc, firstPixel, lastPixel)) {
          // skip invalid pixels
          if (digi_view[i].moduleId() == ::pixelClustering::invalidModuleId)
            continue;
          uint16_t x = digi_view[i].xx();
          uint16_t y = digi_view[i].yy();
          uint32_t pos = module * TrackerTraits::numRowsInModule * TrackerTraits::numColsInModule +
                         y * TrackerTraits::numColsInModule + x;
          if constexpr (isPhase2) {
            images[pos].id() = static_cast<uint16_t>(i - firstPixel);
          } else {
            // check for duplicate pixels only for the Phase 1 detector
            if (alpaka::atomicCas(acc, &images[pos].id(), kEmptyPixel, static_cast<uint32_t>(i - firstPixel)) !=
                kEmptyPixel) {
              digi_view[i].moduleId() = ::pixelClustering::invalidModuleId;
              digi_view[i].rawIdArr() = 0;
            }
          }
        }

        // ensure that all threads have completed filling the image
        alpaka::syncBlockThreads(acc);

        // for each pixel, look at its 8 neighbours; when two valid pixels
        // within +/- 1 in x or y are found, set their id to the minimum;
        // after the loop, all the pixel in each cluster should have the id
        // equal to the lowest pixel in the cluster (clus[i] == i).
        bool more = true;
        while (alpaka::syncBlockThreadsPredicate<alpaka::BlockOr>(acc, more)) {
          more = false;
          for (uint32_t i : cms::alpakatools::independent_group_elements(
                   acc, TrackerTraits::numRowsInModule * TrackerTraits::numColsInModule)) {
            uint32_t pos = module * TrackerTraits::numRowsInModule * TrackerTraits::numColsInModule + i;

            // do not update empty pixels
            if (images[pos].id() == kEmptyPixel) {
              continue;
            }

            int16_t y = i / TrackerTraits::numColsInModule;
            int16_t x = i % TrackerTraits::numColsInModule;
            int16_t x_min = alpaka::math::max(acc, x - 1, 0);
            int16_t y_min = alpaka::math::max(acc, y - 1, 0);
            int16_t x_max = alpaka::math::min(acc, x + 1, TrackerTraits::numColsInModule - 1);
            int16_t y_max = alpaka::math::min(acc, y + 1, TrackerTraits::numRowsInModule - 1);

            // any valid value is smaller than kEmptyPixel
            uint32_t tmp = kEmptyPixel;
            for (int16_t iy = y_min; iy <= y_max; ++iy) {
              for (int16_t ix = x_min; ix <= x_max; ++ix) {
                uint32_t ipos = module * TrackerTraits::numRowsInModule * TrackerTraits::numColsInModule +
                                iy * TrackerTraits::numColsInModule + ix;
                // potential race condition (here: reading images[i], line 292: writing images[i])
                // however, at most this skips ahead one update, so the result should still be ok
                uint32_t val = images[ipos].id();
                tmp = alpaka::math::min(acc, tmp, val);
              }
            }
            if (images[pos].id() != tmp) {
              ALPAKA_ASSERT_ACC(tmp != kEmptyPixel);
              /* should nver happen 
              if (tmp == kEmptyPixel) {
                printf("invalid update at x=%d, y=%d, id=%d\n", x, y, images[pos].id());
                printf("  "); for (int16_t iy = y_min; iy < y_max; ++iy) { printf("______  "); }; printf("\n");
                for (int16_t ix = x_min; ix < x_max; ++ix) {
                  printf("[ ");
                  for (int16_t iy = y_min; iy < y_max; ++iy) {
                    uint32_t ipos = module * TrackerTraits::numRowsInModule * TrackerTraits::numColsInModule + iy * TrackerTraits::numColsInModule + ix;
                    int32_t val = images[ipos].id();
                    printf("%6d  ", val);
                  }
                  printf(" ]\n");
                }
                printf("  "); for (int16_t iy = y_min; iy < y_max; ++iy) { printf("^^^^^^  "); }; printf("\n");
                printf("\n");
              }
              */
              images[pos].id() = tmp;
              more = true;
            }
          }  // pixel loop
        }  // end while

        // copy the cluster id back into the digi_view
        for (uint32_t i : cms::alpakatools::independent_group_elements(acc, firstPixel, lastPixel)) {
          // skip invalid or duplicate pixels
          if (digi_view[i].moduleId() == ::pixelClustering::invalidModuleId)
            continue;
          uint16_t x = digi_view[i].xx();
          uint16_t y = digi_view[i].yy();
          uint32_t pos = module * TrackerTraits::numRowsInModule * TrackerTraits::numColsInModule +
                         y * TrackerTraits::numColsInModule + x;
          digi_view[i].clus() = images[pos].id() + firstPixel;
        }

        auto& foundClusters = alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
        foundClusters = 0;
        alpaka::syncBlockThreads(acc);

        // find the number of different clusters, identified by a pixels with clus[i] == i;
        // mark these pixels with a negative id.
        for (uint32_t i : cms::alpakatools::independent_group_elements(acc, firstPixel, lastPixel)) {
          // skip invalid or duplicate pixels
          if (digi_view[i].moduleId() == ::pixelClustering::invalidModuleId)
            continue;
          if (digi_view[i].clus() == static_cast<int>(i)) {
            auto old = alpaka::atomicInc(acc, &foundClusters, 0xffffffff, alpaka::hierarchy::Threads{});
            digi_view[i].clus() = -(old + 1);
          }
        }
        alpaka::syncBlockThreads(acc);

        // propagate the negative id to all the pixels in the cluster.
        for (uint32_t i : cms::alpakatools::independent_group_elements(acc, firstPixel, lastPixel)) {
          // skip invalid pixels
          if (digi_view[i].moduleId() == ::pixelClustering::invalidModuleId)
            continue;
          if (digi_view[i].clus() >= 0) {
            // mark each pixel in a cluster with the same id as the first one
            digi_view[i].clus() = digi_view[digi_view[i].clus()].clus();
          }
        }
        alpaka::syncBlockThreads(acc);

        // adjust the cluster id to be a positive value starting from 0
        for (uint32_t i : cms::alpakatools::independent_group_elements(acc, firstPixel, lastPixel)) {
          if (digi_view[i].moduleId() == ::pixelClustering::invalidModuleId) {
            // mark invalid pixels with an invalid cluster index
            digi_view[i].clus() = ::pixelClustering::invalidClusterId;
          } else {
            digi_view[i].clus() = -digi_view[i].clus() - 1;
          }
        }
        alpaka::syncBlockThreads(acc);

        if (cms::alpakatools::once_per_block(acc)) {
          clus_view[thisModuleId].clusInModule() = foundClusters;
          clus_view[module].moduleId() = thisModuleId;
#ifdef GPU_DEBUG
          if (foundClusters > gMaxHit) {
            gMaxHit = foundClusters;
            if (foundClusters > 8)
              printf("max hit %d in %d\n", foundClusters, thisModuleId);
          }
          if (thisModuleId % 100 == 1)
            printf("%d clusters in module %d\n", foundClusters, thisModuleId);
#endif
        }
      }  // module loop
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::pixelClustering

#endif  // plugin_SiPixelClusterizer_alpaka_PixelClustering.h
