#ifndef DataFormats_BeamSpotAlpaka_interface_alpaka_BeamSpotAlpaka_h
#define DataFormats_BeamSpotAlpaka_interface_alpaka_BeamSpotAlpaka_h

#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "DataFormats/BeamSpot/interface/BeamSpotPOD.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class BeamSpotAlpaka {
  public:
    // default constructor, required by cms::alpakatools::Product<BeamSpotAlpaka>
    BeamSpotAlpaka() = default;

    // constructor that allocates cached device memory on the given CUDA stream
    BeamSpotAlpaka(Queue& queue) : data_d_{cms::alpakatools::make_device_buffer<BeamSpotPOD>(queue)} {}

    // movable, non-copiable
    BeamSpotAlpaka(BeamSpotAlpaka const&) = delete;
    BeamSpotAlpaka(BeamSpotAlpaka&&) = default;
    BeamSpotAlpaka& operator=(BeamSpotAlpaka const&) = delete;
    BeamSpotAlpaka& operator=(BeamSpotAlpaka&&) = default;

    BeamSpotPOD* data() { return data_d_->data(); }
    BeamSpotPOD const* data() const { return data_d_->data(); }

    cms::alpakatools::device_buffer<Device, BeamSpotPOD>& ptr() { return *data_d_; }
    cms::alpakatools::device_buffer<Device, BeamSpotPOD> const& ptr() const { return *data_d_; }

  private:
    std::optional<cms::alpakatools::device_buffer<Device, BeamSpotPOD>> data_d_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_BeamSpotAlpaka_interface_alpaka_BeamSpotAlpaka_h
