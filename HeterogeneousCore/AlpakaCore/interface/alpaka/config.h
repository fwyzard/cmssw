#pragma once

#include <type_traits>

#include <alpaka/alpaka.hpp>

namespace alpaka_common {

  // common types and dimensions
  using Idx = uint32_t;
  using Extent = uint32_t;
  using Offsets = Extent;

  using Dim0D = alpaka::DimInt<0u>;
  using Dim1D = alpaka::DimInt<1u>;
  using Dim2D = alpaka::DimInt<2u>;
  using Dim3D = alpaka::DimInt<3u>;

  template <typename TDim>
  using Vec = alpaka::Vec<TDim, Idx>;
  using Vec1D = Vec<Dim1D>;
  using Vec2D = Vec<Dim2D>;
  using Vec3D = Vec<Dim3D>;
  using Scalar = Vec<Dim0D>;

  template <typename TDim>
  using WorkDiv = alpaka::WorkDivMembers<TDim, Idx>;
  using WorkDiv1D = WorkDiv<Dim1D>;
  using WorkDiv2D = WorkDiv<Dim2D>;
  using WorkDiv3D = WorkDiv<Dim3D>;

  // host types
  using DevHost = alpaka::DevCpu;
  using PltfHost = alpaka::Pltf<DevHost>;

}  // namespace alpaka_common

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
namespace alpaka_cuda_async {
  using namespace alpaka_common;

  using Platform = alpaka::PltfCudaRt;
  using Device = alpaka::DevCudaRt;
  using Queue = alpaka::QueueCudaRtNonBlocking;
  using Event = alpaka::EventCudaRt;

  template <typename TDim>
  using Acc = alpaka::AccGpuCudaRt<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;
  using Acc2D = Acc<Dim2D>;
  using Acc3D = Acc<Dim3D>;

}  // namespace alpaka_cuda_async

#ifdef ALPAKA_ACCELERATOR_NAMESPACE
#define ALPAKA_DUPLICATE_NAMESPACE
#else
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_cuda_async
#endif

#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
namespace alpaka_hip_async {
  using namespace alpaka_common;

  using Platform = alpaka::PltfHipRt;
  using Device = alpaka::DevHipRt;
  using Queue = alpaka::QueueHipRtNonBlocking;
  using Event = alpaka::EventHipRt;

  template <typename TDim>
  using Acc = alpaka::AccGpuHipRt<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;
  using Acc2D = Acc<Dim2D>;
  using Acc3D = Acc<Dim3D>;

}  // namespace alpaka_hip_async

#ifdef ALPAKA_ACCELERATOR_NAMESPACE
#define ALPAKA_DUPLICATE_NAMESPACE
#else
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_hip_async
#endif

#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
namespace alpaka_serial_sync {
  using namespace alpaka_common;

  using Platform = alpaka::PltfCpu;
  using Device = alpaka::DevCpu;
  using Queue = alpaka::QueueCpuBlocking;
  using Event = alpaka::EventCpu;

  template <typename TDim>
  using Acc = alpaka::AccCpuSerial<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;
  using Acc2D = Acc<Dim2D>;
  using Acc3D = Acc<Dim3D>;

}  // namespace alpaka_serial_sync

#ifdef ALPAKA_ACCELERATOR_NAMESPACE
#define ALPAKA_DUPLICATE_NAMESPACE
#else
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_serial_sync
#endif

#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
namespace alpaka_tbb_async {
  using namespace alpaka_common;

  using Platform = alpaka::PltfCpu;
  using Device = alpaka::DevCpu;
  using Queue = alpaka::QueueCpuNonBlocking;
  using Event = alpaka::EventCpu;

  template <typename TDim>
  using Acc = alpaka::AccCpuTbbBlocks<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;
  using Acc2D = Acc<Dim2D>;
  using Acc3D = Acc<Dim3D>;

}  // namespace alpaka_tbb_async

#ifdef ALPAKA_ACCELERATOR_NAMESPACE
#define ALPAKA_DUPLICATE_NAMESPACE
#else
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_tbb_async
#endif

#endif  // ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

#if not defined ALPAKA_ACCELERATOR_NAMESPACE or defined ALPAKA_DUPLICATE_NAMESPACE
#error Exactly one must be defined among: ALPAKA_ACC_GPU_CUDA_ENABLED, ALPAKA_ACC_GPU_HIP_ENABLED, ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED, ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED.
#endif
