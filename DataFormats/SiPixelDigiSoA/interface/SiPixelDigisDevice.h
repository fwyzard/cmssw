#ifndef DataFormats_SiPixelDigiSoA_interface_SiPixelDigisDevice_h
#define DataFormats_SiPixelDigiSoA_interface_SiPixelDigisDevice_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisHost.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisLayout.h"

template <typename TDev>
class SiPixelDigisDevice : public PortableDeviceCollection<SiPixelDigisLayout<>, TDev> {
public:
  SiPixelDigisDevice() = default;
  template <typename TQueue>
  explicit SiPixelDigisDevice(size_t maxFedWords, TQueue queue)
      : PortableDeviceCollection<SiPixelDigisLayout<>, TDev>(maxFedWords + 1, queue) {}
  ~SiPixelDigisDevice() = default;

  // Constructor which specifies the SoA size
  explicit SiPixelDigisDevice(size_t maxFedWords, TDev const &device)
      : PortableDeviceCollection<SiPixelDigisLayout<>, TDev>(maxFedWords + 1, device) {}

  SiPixelDigisDevice(SiPixelDigisDevice &&) = default;
  SiPixelDigisDevice &operator=(SiPixelDigisDevice &&) = default;

  void setNModulesDigis(uint32_t nModules, uint32_t nDigis) {
    nModules_h = nModules;
    nDigis_h = nDigis;
  }

  uint32_t nModules() const { return nModules_h; }
  uint32_t nDigis() const { return nDigis_h; }

private:
  uint32_t nModules_h = 0;
  uint32_t nDigis_h = 0;
};

#endif  // DataFormats_SiPixelDigiSoA_interface_SiPixelDigisDevice_h