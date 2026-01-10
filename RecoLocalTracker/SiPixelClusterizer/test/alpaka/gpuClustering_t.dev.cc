#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <set>
#include <vector>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelClusterThresholds.h"

// local includes, for testing only
#include "RecoLocalTracker/SiPixelClusterizer/plugins/alpaka/ClusterChargeCut.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/alpaka/PixelClustering.h"

using namespace cms::alpakatools;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;
using namespace ALPAKA_ACCELERATOR_NAMESPACE::pixelClustering;
// FIXME migrate to phase1PixelTopology ?
using namespace ::pixelClustering;
//using pixelTopology::Phase1;

int main(void) {
  // get the list of devices on the current platform
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "the test will be skipped.\n";
    exit(EXIT_FAILURE);
  }

  // loop on the available devices
  for (Device device : devices) {
    Queue queue{device};

    constexpr int numElements = 256 * maxNumModules;
    const SiPixelClusterThresholds clusterThresholds(
        clusterThresholdLayerOne, clusterThresholdOtherLayers, 0.f, 0.f, 0.f, 0.f);

    auto h_raw = make_host_buffer<uint32_t[]>(queue, numElements);
    auto h_id = make_host_buffer<uint16_t[]>(queue, numElements);
    auto h_x = make_host_buffer<uint16_t[]>(queue, numElements);
    auto h_y = make_host_buffer<uint16_t[]>(queue, numElements);
    auto h_adc = make_host_buffer<uint16_t[]>(queue, numElements);
    auto h_clus = make_host_buffer<int[]>(queue, numElements);
    auto h_clusInModule = make_host_buffer<uint32_t[]>(queue, maxNumModules);

    auto d_raw = make_device_buffer<uint32_t[]>(queue, numElements);
    auto d_id = make_device_buffer<uint16_t[]>(queue, numElements);
    auto d_x = make_device_buffer<uint16_t[]>(queue, numElements);
    auto d_y = make_device_buffer<uint16_t[]>(queue, numElements);
    auto d_adc = make_device_buffer<uint16_t[]>(queue, numElements);
    auto d_clus = make_device_buffer<int[]>(queue, numElements);
    auto d_moduleStart = make_device_buffer<uint32_t[]>(queue, maxNumModules + 1);
    auto d_clusInModule = make_device_buffer<uint32_t[]>(queue, maxNumModules);
    auto d_moduleId = make_device_buffer<uint32_t[]>(queue, maxNumModules);

    // later random number
    int n = 0;
    int ncl = 0;
    int y[10] = {5, 7, 9, 1, 3, 0, 4, 8, 2, 6};

    auto generateClusters = [&](int kn) {
      auto addBigNoise = 1 == kn % 2;
      if (addBigNoise) {
        constexpr int MaxPixels = 1000;
        int id = 666;
        for (int x = 0; x < 140; x += 3) {
          for (int yy = 0; yy < 400; yy += 3) {
            h_id[n] = id;
            h_x[n] = x;
            h_y[n] = yy;
            h_adc[n] = 1000;
            ++n;
            ++ncl;
            if (MaxPixels <= ncl)
              break;
          }
          if (MaxPixels <= ncl)
            break;
        }
      }

      {
        // isolated
        int id = 42;
        int x = 10;
        ++ncl;
        h_id[n] = id;
        h_x[n] = x;
        h_y[n] = x;
        h_adc[n] = kn == 0 ? 100 : 5000;
        ++n;

        // first column
        ++ncl;
        h_id[n] = id;
        h_x[n] = x;
        h_y[n] = 0;
        h_adc[n] = 5000;
        ++n;
        // first columns
        ++ncl;
        h_id[n] = id;
        h_x[n] = x + 80;
        h_y[n] = 2;
        h_adc[n] = 5000;
        ++n;
        h_id[n] = id;
        h_x[n] = x + 80;
        h_y[n] = 1;
        h_adc[n] = 5000;
        ++n;

        // last column
        ++ncl;
        h_id[n] = id;
        h_x[n] = x;
        h_y[n] = 415;
        h_adc[n] = 5000;
        ++n;
        // last columns
        ++ncl;
        h_id[n] = id;
        h_x[n] = x + 80;
        h_y[n] = 415;
        h_adc[n] = 2500;
        ++n;
        h_id[n] = id;
        h_x[n] = x + 80;
        h_y[n] = 414;
        h_adc[n] = 2500;
        ++n;

        // diagonal
        ++ncl;
        for (int x = 20; x < 25; ++x) {
          h_id[n] = id;
          h_x[n] = x;
          h_y[n] = x;
          h_adc[n] = 1000;
          ++n;
        }
        ++ncl;
        // reversed
        for (int x = 45; x > 40; --x) {
          h_id[n] = id;
          h_x[n] = x;
          h_y[n] = x;
          h_adc[n] = 1000;
          ++n;
        }
        ++ncl;
        h_id[n++] = invalidModuleId;  // error
        // messy
        int xx[5] = {21, 25, 23, 24, 22};
        for (int k = 0; k < 5; ++k) {
          h_id[n] = id;
          h_x[n] = xx[k];
          h_y[n] = 20 + xx[k];
          h_adc[n] = 1000;
          ++n;
        }
        // holes
        ++ncl;
        for (int k = 0; k < 5; ++k) {
          h_id[n] = id;
          h_x[n] = xx[k];
          h_y[n] = 100;
          h_adc[n] = kn == 2 ? 100 : 1000;
          ++n;
          if (xx[k] % 2 == 0) {
            h_id[n] = id;
            h_x[n] = xx[k];
            h_y[n] = 101;
            h_adc[n] = 1000;
            ++n;
          }
        }
      }
      {
        // id == 0 (make sure it works!)
        int id = 0;
        int x = 10;
        ++ncl;
        h_id[n] = id;
        h_x[n] = x;
        h_y[n] = x;
        h_adc[n] = 5000;
        ++n;
      }
      // all odd id
      for (int id = 11; id <= 1800; id += 2) {
        if ((id / 20) % 2)
          h_id[n++] = invalidModuleId;  // error
        for (int x = 0; x < 40; x += 4) {
          ++ncl;
          if ((id / 10) % 2) {
            for (int k = 0; k < 10; ++k) {
              h_id[n] = id;
              h_x[n] = x;
              h_y[n] = x + y[k];
              h_adc[n] = 100;
              ++n;
              h_id[n] = id;
              h_x[n] = x + 1;
              h_y[n] = x + y[k] + 2;
              h_adc[n] = 1000;
              ++n;
            }
          } else {
            for (int k = 0; k < 10; ++k) {
              h_id[n] = id;
              h_x[n] = x;
              h_y[n] = x + y[9 - k];
              h_adc[n] = kn == 2 ? 10 : 1000;
              ++n;
              if (y[k] == 3)
                continue;  // hole
              if (id == 51) {
                h_id[n++] = invalidModuleId;
                h_id[n++] = invalidModuleId;
              }  // error
              h_id[n] = id;
              h_x[n] = x + 1;
              h_y[n] = x + y[k] + 2;
              h_adc[n] = kn == 2 ? 10 : 1000;
              ++n;
            }
          }
        }
      }
    };  // end lambda

    for (auto kkk = 0; kkk < 5; ++kkk) {
      n = 0;
      ncl = 0;
      generateClusters(kkk);

      std::cout << "created " << n << " digis in " << ncl << " clusters" << std::endl;
      assert(n <= numElements);

      auto nModules = make_host_buffer<uint32_t>(queue);
      *nModules = 0;

      alpaka::fill(queue, d_moduleStart, 0);
      alpaka::memcpy(queue, d_id, h_id);
      alpaka::memcpy(queue, d_x, h_x);
      alpaka::memcpy(queue, d_y, h_y);
      alpaka::memcpy(queue, d_adc, h_adc);

      // Launch CUDA Kernels
      int threadsPerBlock = (kkk == 5) ? 512 : ((kkk == 3) ? 128 : 256);
      int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
      std::cout << "CUDA countModules kernel launch with " << blocksPerGrid << " blocks of " << threadsPerBlock
                << " threads\n";

      /*
      alpaka::exec<Acc1D>(queue,
                          make_workdiv<Acc1D>(blocksPerGrid, threadsPerBlock),
                          CountModules<>{},
                          d_id.data(),
                          d_moduleStart.data(),
                          d_clus.data(),
                          n);
      alpaka::wait(queue);
      */

      blocksPerGrid = maxNumModules;

      std::cout << "CUDA findModules kernel launch with " << blocksPerGrid << " blocks of " << threadsPerBlock
                << " threads\n";
      alpaka::fill(queue, d_clusInModule, 0);

      /*
      alpaka::exec<Acc1D>(queue,
                          make_workdiv<Acc1D>(blocksPerGrid, threadsPerBlock),
                          findClus<Phase1>,
                          d_raw.data(),
                          d_id.data(),
                          d_x.data(),
                          d_y.data(),
                          d_moduleStart.data(),
                          d_clusInModule.data(),
                          d_moduleId.data(),
                          d_clus.data(),
                          n);
      alpaka::wait(queue);
      */
      alpaka::memcpy(queue, nModules, make_device_view<uint32_t>(queue, d_moduleStart.data()[0]));
      auto h_moduleId = make_host_buffer<uint32_t[]>(queue, *nModules);
      alpaka::memcpy(queue, h_moduleId, d_moduleId, *nModules);
      alpaka::memcpy(queue, h_clusInModule, d_clusInModule);

      std::cout << "before charge cut found "
                << std::accumulate(h_clusInModule.data(), h_clusInModule.data() + maxNumModules, 0) << " clusters"
                << std::endl;
      for (auto i = maxNumModules; i > 0; i--)
        if (h_clusInModule[i - 1] > 0) {
          std::cout << "last module is " << i - 1 << ' ' << h_clusInModule[i - 1] << std::endl;
          break;
        }
      if (ncl != std::accumulate(h_clusInModule.data(), h_clusInModule.data() + maxNumModules, 0))
        std::cout << "ERROR!!!!! wrong number of cluster found" << std::endl;

      /*
      alpaka::exec<Acc1D>(queue,
                          make_workdiv<Acc1D>(blocksPerGrid, threadsPerBlock),
                          clusterChargeCut<Phase1>,
                          clusterThresholds,
                          d_id.data(),
                          d_adc.data(),
                          d_moduleStart.data(),
                          d_clusInModule.data(),
                          d_moduleId.data(),
                          d_clus.data(),
                          n);
      alpaka::wait(queue);
      */

      alpaka::memcpy(queue, h_id, d_id);
      alpaka::memcpy(queue, h_clus, d_clus);
      alpaka::wait(queue);

      std::cout << "found " << *nModules << " Modules active" << std::endl;

      std::set<unsigned int> clids;
      for (int i = 0; i < n; ++i) {
        assert(h_id[i] != 666);  // only noise
        if (h_id[i] == invalidModuleId)
          continue;
        assert(h_clus[i] >= 0);
        assert(h_clus[i] < static_cast<int>(h_clusInModule[h_id[i]]));
        clids.insert(h_id[i] * 1000 + h_clus[i]);
        // clids.insert(h_clus[i]);
      }

      // verify no hole in numbering
      auto p = clids.begin();
      auto cmid = (*p) / 1000;
      assert(0 == (*p) % 1000);
      auto c = p;
      ++c;
      std::cout << "first clusters " << *p << ' ' << *c << ' ' << h_clusInModule[cmid] << ' '
                << h_clusInModule[(*c) / 1000] << std::endl;
      std::cout << "last cluster " << *clids.rbegin() << ' ' << h_clusInModule[(*clids.rbegin()) / 1000] << std::endl;
      for (; c != clids.end(); ++c) {
        auto cc = *c;
        auto pp = *p;
        auto mid = cc / 1000;
        auto pnc = pp % 1000;
        auto nc = cc % 1000;
        if (mid != cmid) {
          assert(0 == cc % 1000);
          assert(h_clusInModule[cmid] - 1 == pp % 1000);
          // if (h_clusInModule[cmid]-1 != pp%1000) std::cout << "error size " << mid << ": "  << h_clusInModule[mid] << ' ' << pp << std::endl;
          cmid = mid;
          p = c;
          continue;
        }
        p = c;
        // assert(nc==pnc+1);
        if (nc != pnc + 1)
          std::cout << "error " << mid << ": " << nc << ' ' << pnc << std::endl;
      }

      std::cout << "found " << std::accumulate(h_clusInModule.data(), h_clusInModule.data() + maxNumModules, 0) << ' '
                << clids.size() << " clusters" << std::endl;
      for (auto i = maxNumModules; i > 0; i--)
        if (h_clusInModule[i - 1] > 0) {
          std::cout << "last module is " << i - 1 << ' ' << h_clusInModule[i - 1] << std::endl;
          break;
        }
      // << " and " << seeds.size() << " seeds" << std::endl;
    }  /// end loop kkk

  }  // end loop on the available devices

  return 0;
}
