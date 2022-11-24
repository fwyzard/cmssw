#ifndef DataFormats_Portable_interface_alpaka_PortableDeviceCollection_h
#define DataFormats_Portable_interface_alpaka_PortableDeviceCollection_h

#include <optional>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/TransferToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  // ... or any other CPU-based accelerators
  template <typename T0, typename T1 = void, typename T2 = void, typename T3 = void, typename T4 = void>
  using PortableCollection = ::PortableHostCollection<T0, T1, T2, T3, T4>;
#else
  template <typename T0, typename T1 = void, typename T2 = void, typename T3 = void, typename T4 = void>
  using PortableCollection = ::PortableDeviceCollection<Device, T0, T1, T2, T3, T4>;
#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace traits {

  // specialise the trait for the device provided by the ALPAKA_ACCELERATOR_NAMESPACE
  template <typename T0, typename T1, typename T2, typename T3, typename T4>
  class PortableCollectionTrait<ALPAKA_ACCELERATOR_NAMESPACE::Device, T0, T1, T2, T3, T4> {
    using CollectionType = ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection<T0, T1, T2, T3, T4>;
  };

}  // namespace traits

namespace cms::alpakatools {
  // TODO: Is this the right place for the specialization? Or should it be in PortableDeviceProduct?
  template <typename T0, typename T1, typename T2, typename T3, typename T4>
  struct TransferToHost<ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection<T0, T1, T2, T3, T4>> {
    using HostDataType = ::PortableHostCollection<T0, T1, T2, T3, T4>;

    template <typename TQueue>
    static HostDataType transferAsync(
        TQueue& queue, ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection<T0, T1, T2, T3, T4> const& deviceData) {
      HostDataType hostData(deviceData.sizes(), queue);
      alpaka::memcpy(queue, hostData.buffer(), deviceData.buffer());
      return hostData;
    }
  };

}  // namespace cms::alpakatools

#endif  // DataFormats_Portable_interface_alpaka_PortableDeviceCollection_h
