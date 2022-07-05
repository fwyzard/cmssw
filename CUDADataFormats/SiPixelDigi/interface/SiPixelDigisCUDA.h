#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h

#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"

#include "DataFormats/SoATemplate/interface/SoALayout.h"

// Host and device layout: data used on both sides and transferred from device to host.
GENERATE_SOA_LAYOUT(SiPixelDigisCUDASOA_H_D_Template,
  SOA_COLUMN(int32_t, clus),
  SOA_COLUMN(uint32_t, pdigi),
  SOA_COLUMN(uint32_t, rawIdArr),
  SOA_COLUMN(uint16_t, adc)
//  SOA_SCALAR(uint32_t, nModules),
//  SOA_SCALAR(uint32_t, nDigis)
)

// Device only data: this will not be transferred to host.
GENERATE_SOA_LAYOUT(SiPixelDigisCUDASOA_DO_Template,
  SOA_COLUMN(uint16_t, xx),
  SOA_COLUMN(uint16_t, yy),
  SOA_COLUMN(uint16_t, moduleInd)
)

using SiPixelDigisCUDASOA_H_D = SiPixelDigisCUDASOA_H_D_Template<>;
using SiPixelDigisCUDASOA_DO = SiPixelDigisCUDASOA_DO_Template<>;

// Device view joining the 2 previous layouts in a single place.
GENERATE_SOA_VIEW(SiPixelDigisCUDASOA_D_View_Template,
  SOA_VIEW_LAYOUT_LIST(
    SOA_VIEW_LAYOUT(SiPixelDigisCUDASOA_H_D, hostDevice),
    SOA_VIEW_LAYOUT(SiPixelDigisCUDASOA_DO, deviceOnly)
  ),
  SOA_VIEW_VALUE_LIST(
    SOA_VIEW_VALUE(hostDevice, clus),
    SOA_VIEW_VALUE(hostDevice, pdigi),
    SOA_VIEW_VALUE(hostDevice, rawIdArr),
    SOA_VIEW_VALUE(hostDevice, adc),
    SOA_VIEW_VALUE(deviceOnly, xx),
    SOA_VIEW_VALUE(deviceOnly, yy),
    SOA_VIEW_VALUE(deviceOnly, moduleInd)
  )
)

using SiPixelDigisCUDASOAView = SiPixelDigisCUDASOA_D_View_Template<>;

// Device view joining the 2 previous layouts in a single place.
GENERATE_SOA_CONST_VIEW(SiPixelDigisCUDASOA_D_View_ConstTemplate,
  SOA_VIEW_LAYOUT_LIST(
    SOA_VIEW_LAYOUT(SiPixelDigisCUDASOA_H_D, hostDevice),
    SOA_VIEW_LAYOUT(SiPixelDigisCUDASOA_DO, deviceOnly)
  ),
  SOA_VIEW_VALUE_LIST(
    SOA_VIEW_VALUE(hostDevice, clus),
    SOA_VIEW_VALUE(hostDevice, pdigi),
    SOA_VIEW_VALUE(hostDevice, rawIdArr),
    SOA_VIEW_VALUE(hostDevice, adc),
    SOA_VIEW_VALUE(deviceOnly, xx),
    SOA_VIEW_VALUE(deviceOnly, yy),
    SOA_VIEW_VALUE(deviceOnly, moduleInd)
  )
)

using SiPixelDigisCUDASOAConstView = SiPixelDigisCUDASOA_D_View_ConstTemplate<>;

//using SiPixelDigisCUDA = PortableCUDAHostCollection<SiPixelDigisCUDASOA>;

class SiPixelDigisCUDA {
public:
  //using StoreType = uint16_t;
  SiPixelDigisCUDA() = default;
  explicit SiPixelDigisCUDA(size_t maxFedWords, cudaStream_t stream);
  ~SiPixelDigisCUDA() = default;

  SiPixelDigisCUDA(const SiPixelDigisCUDA &) = delete;
  SiPixelDigisCUDA &operator=(const SiPixelDigisCUDA &) = delete;
  SiPixelDigisCUDA(SiPixelDigisCUDA &&) = default;
  SiPixelDigisCUDA &operator=(SiPixelDigisCUDA &&) = default;

  void setNModulesDigis(uint32_t nModules, uint32_t nDigis) {
    nModules_h = nModules;
    nDigis_h = nDigis;
  }

  uint32_t nModules() const { return nModules_h; }
  uint32_t nDigis() const { return nDigis_h; }

  cms::cuda::host::unique_ptr<std::byte[]> copyAllToHostAsync(cudaStream_t stream) const;

  SiPixelDigisCUDASOAView view() { return m_view; }
  SiPixelDigisCUDASOAConstView const view() const { return m_constView; }

private:
  // These are consumed by downstream device code
  cms::cuda::device::unique_ptr<std::byte[]> m_buffer;
  SiPixelDigisCUDASOA_H_D m_hostDeviceLayout;
  SiPixelDigisCUDASOA_DO  m_deviceOnlyLayout;
  SiPixelDigisCUDASOAView m_view;
  SiPixelDigisCUDASOAConstView m_constView;
  

  uint32_t nModules_h = 0;
  uint32_t nDigis_h = 0;
};

#endif  // CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h
