#include <cassert>

#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

SiPixelDigisCUDA::SiPixelDigisCUDA(size_t maxFedWords, cudaStream_t stream)
    : m_buffer(cms::cuda::make_device_unique<std::byte[]>(
          SiPixelDigisCUDASOA_H_D::computeDataSize(maxFedWords) +
          SiPixelDigisCUDASOA_DO::computeDataSize(maxFedWords),
          stream)),
      m_hostDeviceLayout(m_buffer.get(), maxFedWords),
      m_deviceOnlyLayout(m_hostDeviceLayout.metadata().nextByte(), maxFedWords),
      m_view(m_hostDeviceLayout, m_deviceOnlyLayout),
      m_constView(m_hostDeviceLayout, m_deviceOnlyLayout) {
  assert(maxFedWords != 0);
}

cms::cuda::host::unique_ptr<std::byte[]> SiPixelDigisCUDA::copyAllToHostAsync(
    cudaStream_t stream) const {
  // Copy to a host buffer the host-device shared part (m_hostDeviceLayout).
  auto ret = cms::cuda::make_host_unique<std::byte[]>(m_hostDeviceLayout.metadata().byteSize(), stream);
  cudaCheck(cudaMemcpyAsync(ret.get(),
                            m_hostDeviceLayout.metadata().data(),
                            m_hostDeviceLayout.metadata().byteSize(),
                            cudaMemcpyDeviceToHost,
                            stream));
  return ret;
}
