#ifndef DataFormats_Vertex_ZVertexSoADevice_H
#define DataFormats_Vertex_ZVertexSoADevice_H

#include <cstdint>

#include <alpaka/alpaka.hpp>
#include "DataFormats/Vertex/interface/ZVertexLayout.h"
#include "DataFormats/Vertex/interface/ZVertexDefinitions.h"
#include "DataFormats/Vertex/interface/alpaka/ZVertexUtilities.h"
#include "DataFormats/Vertex/interface/ZVertexSoAHost.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <int32_t S>
  class ZVertexSoADevice : public PortableCollection<ZVertexSoAHeterogeneousLayout<>> {
  public:
    ZVertexSoADevice() = default;  // cms::alpakatools::Product needs this

    // Constructor which specifies the SoA size
    explicit ZVertexSoADevice(Queue queue) : PortableCollection<ZVertexSoAHeterogeneousLayout<>>(S, queue) {}

    // Constructor which specifies the SoA size
    explicit ZVertexSoADevice(Device const& device) : PortableCollection<ZVertexSoAHeterogeneousLayout<>>(S, device) {}
  };

  using namespace ::zVertex;
  using ZVertexDevice = ZVertexSoADevice<MAXTRACKS>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <>
  struct CopyToHost<ALPAKA_ACCELERATOR_NAMESPACE::ZVertexDevice> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, ALPAKA_ACCELERATOR_NAMESPACE::ZVertexDevice const& deviceData) {
      ZVertexHost hostData(queue);
      alpaka::memcpy(queue, hostData.buffer(), deviceData.buffer());
      return hostData;
    }
  };
}  // namespace cms::alpakatools

#endif  // DataFormats_Vertex_ZVertexSoADevice_H
