/**
   Simple test for the pixelTrack::TrackSoA data structure
   which inherits from PortableDeviceCollection.

   Creates an instance of the class (automatically allocates
   memory on device), passes the view of the SoA data to
   the CUDA kernels which:
   - Fill the SoA with data.
   - Verify that the data written is correct.

   Then, the SoA data are copied back to Host, where
   a temporary host-side view (tmp_view) is created using
   the same Layout to access the data on host and print it.
 */

#include <alpaka/alpaka.hpp>
#include <unistd.h>
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitSoADevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsLayout.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitSoAHost.h"
#include "DataFormats/Track/interface/PixelTrackLayout.h"
#include "DataFormats/Track/interface/alpaka/TrackSoADevice.h"
#include "HeterogeneousCore/AlpakaUtilities/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

using namespace std;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

using Tuples = typename TrackSoA<pixelTopology::Phase1>::HitContainer;
using TupleMultiplicity = cms::alpakatools::OneToManyAssoc<typename pixelTopology::Phase1::tindex_type,
                                                           pixelTopology::Phase1::maxHitsOnTrack + 1,
                                                           pixelTopology::Phase1::maxNumberOfTuples>;

template <int N>
using Matrix3xNd = Eigen::Matrix<double, 3, N>;
template <int N>
using Matrix6xNf = Eigen::Matrix<float, 6, N>;
using Vector4d = Eigen::Vector4d;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  void runKernels(Tuples const *foundNtuplets,
                  TupleMultiplicity const *tupleMultiplicity,
                  TrackingRecHitAlpakaSoAConstView<pixelTopology::Phase1> hh,
                  typename pixelTopology::Phase1::tindex_type *ptkids,
                  double *phits,
                  float *phits_ge,
                  double *pfast_fit,
                  uint32_t nHitsL,
                  uint32_t nHitsH,
                  int32_t offset,
                  Queue &queue);
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

int main() {
  const auto host = cms::alpakatools::host();
  const auto device = cms::alpakatools::devices<Platform>()[0];
  Queue queue(device);

  // Inner scope to deallocate memory before destroying the stream
  {
    int maxNumberOfConcurrentFits_ = 10;

    TrackSoADevice<pixelTopology::Phase1> tracks_d;
    Tuples *tuples_d = &tracks_d.view().hitIndices();

    // auto tuplesMult_d =
    //     cms::alpakatools::make_device_buffer<typename TupleMultiplicity>(queue, maxNumberOfConcurrentFits_);
    // const Tuples *tuples_d;
    const TupleMultiplicity *tuplesMult_d;

    const TrackingRecHitAlpakaSoA<pixelTopology::Phase1>::ParamsOnDevice *cpeParams;
    uint32_t const oi = 1;
    TrackingRecHitAlpakaDevice<pixelTopology::Phase1> hits_d(1, 1, cpeParams, &oi, queue);

    auto tkidGPU = cms::alpakatools::make_device_buffer<typename pixelTopology::Phase1::tindex_type[]>(
        queue, maxNumberOfConcurrentFits_);
    auto hitsGPU = cms::alpakatools::make_device_buffer<double[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(Matrix3xNd<6>) / sizeof(double));
    auto hits_geGPU = cms::alpakatools::make_device_buffer<float[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(Matrix6xNf<6>) / sizeof(float));
    auto fast_fit_resultsGPU = cms::alpakatools::make_device_buffer<double[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(Vector4d) / sizeof(double));

    runKernels(tuples_d,
               tuplesMult_d,
               hits_d.view(),
               tkidGPU.data(),
               hitsGPU.data(),
               hits_geGPU.data(),
               fast_fit_resultsGPU.data(),
               3,
               3,
               1,
               queue);

    // Instantate tracks on host. This is where the data will be
    // copied to from device.
    // TrackSoAHost<pixelTopology::Phase1> tracks_h(queue);

    // std::cout << tracks_h.view().metadata().size() << std::endl;
    // alpaka::memcpy(queue, tracks_h.buffer(), tracks_d.const_buffer());
    // alpaka::wait(queue);
  }

  return 0;
}
