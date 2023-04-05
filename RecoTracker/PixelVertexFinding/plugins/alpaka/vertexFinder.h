#ifndef RecoPixelVertexing_PixelVertexFinding_vertexFinder_h
#define RecoPixelVertexing_PixelVertexFinding_vertexFinder_h

#include <cstddef>
#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "DataFormats/Track/interface/alpaka/PixelTrackUtilities.h"
#include "DataFormats/Vertex/interface/ZVertexSoAHost.h"
#include "DataFormats/Vertex/interface/ZVertexLayout.h"
#include "DataFormats/Vertex/interface/alpaka/ZVertexSoADevice.h"
#include "DataFormats/Vertex/interface/alpaka/ZVertexUtilities.h"
#include "../PixelVertexWorkSpaceLayout.h"
#include "PixelVertexWorkSpaceUtilitiesAlpaka.h"
#include "PixelVertexWorkSpaceSoADeviceAlpaka.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace vertexFinder {
    using namespace cms::alpakatools;
    using VtxSoAView = ::zVertex::ZVertexSoAView;
    using WsSoAView = ::vertexFinder::workSpace::PixelVertexWorkSpaceSoAView;

    class init {
    public:
      template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
      ALPAKA_FN_ACC void operator()(const TAcc &acc, VtxSoAView pdata, WsSoAView pws) const {
        pdata.nvFinal() = 0;  // initialization
        vertexFinder::workSpace::utilities::init(pws);
      }
    };

    template <typename TrackerTraits>
    class Producer {
      using TkSoAConstView = TrackSoAConstView<TrackerTraits>;

    public:
      Producer(bool oneKernel,
               bool useDensity,
               bool useDBSCAN,
               bool useIterative,
               int iminT,      // min number of neighbours to be "core"
               float ieps,     // max absolute distance to cluster
               float ierrmax,  // max error to be "seed"
               float ichi2max  // max normalized distance to cluster
               )
          : oneKernel_(oneKernel && !(useDBSCAN || useIterative)),
            useDensity_(useDensity),
            useDBSCAN_(useDBSCAN),
            useIterative_(useIterative),
            minT(iminT),
            eps(ieps),
            errmax(ierrmax),
            chi2max(ichi2max) {}

      ~Producer() = default;

      ZVertexDevice makeAsync(Queue &queue, const TkSoAConstView &tracks_view, float ptMin, float ptMax) const;

    private:
      const bool oneKernel_;
      const bool useDensity_;
      const bool useDBSCAN_;
      const bool useIterative_;

      int minT;       // min number of neighbours to be "core"
      float eps;      // max absolute distance to cluster
      float errmax;   // max error to be "seed"
      float chi2max;  // max normalized distance to cluster
    };

  }  // namespace vertexFinder
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif
