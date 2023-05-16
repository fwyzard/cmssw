#ifndef DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsDevice_h
#define DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsDevice_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
// #include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsLayout.h"
// #include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsHost.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "HeterogeneousCore/AlpakaInterface/interface/SimpleVector.h"

template <typename TDev>
class SiPixelDigiErrorsDevice : public PortableDeviceCollection<SiPixelDigiErrorsLayout<>, TDev> {
public:
  SiPixelDigiErrorsDevice() = default;
  template <typename TQueue>
  explicit SiPixelDigiErrorsDevice(size_t maxFedWords, TQueue queue)
      : PortableDeviceCollection<SiPixelDigiErrorsLayout<>, TDev>(maxFedWords, queue), maxFedWords_(maxFedWords) {
    printf("SiPixelDigiErrorsDevice");
    // data_d = cms::alpakatools::make_device_buffer<SiPixelErrorCompact[]>(queue, maxFedWords);
    // error_d = cms::alpakatools::make_device_buffer<cms::alpakatools::SimpleVector<SiPixelErrorCompact>>(queue);
    // (*error_d).data()->construct(maxFedWords, data_d->data());
    this->view().pixelErrorsVec().set_data(this->view().pixelErrors());
    ALPAKA_ASSERT_OFFLOAD(this->view().pixelErrorsVec().empty());
    ALPAKA_ASSERT_OFFLOAD(this->view().pixelErrorsVec().capacity() == static_cast<int>(maxFedWords));
    printf("ok SiPixelDigiErrorsDevice");
  }
  ~SiPixelDigiErrorsDevice() = default;

  // Constructor which specifies the SoA size
  explicit SiPixelDigiErrorsDevice(size_t maxFedWords, TDev const& device)
      : PortableDeviceCollection<SiPixelDigiErrorsLayout<>, TDev>(maxFedWords, device) {}

  SiPixelDigiErrorsDevice(SiPixelDigiErrorsDevice&&) = default;
  SiPixelDigiErrorsDevice& operator=(SiPixelDigiErrorsDevice&&) = default;

  cms::alpakatools::SimpleVector<SiPixelErrorCompact>* error() { return (&this->view().pixelErrorsVec()); }
  cms::alpakatools::SimpleVector<SiPixelErrorCompact> const* error() const { return (&this->view().pixelErrorsVec()); }
  cms::alpakatools::SimpleVector<SiPixelErrorCompact> const* c_error() const {
    return (&this->view().pixelErrorsVec());
  }

  auto& error_vector() const { return (this->view().pixelErrorsVec()); }
  auto& error_data() const { return (*this->view().pixelErrors()); }
  auto maxFedWords() const { return maxFedWords_; }

private:
  int maxFedWords_;
  // std::optional<cms::alpakatools::device_buffer<TDev, SiPixelErrorCompact[]>> data_d;
  // std::optional<cms::alpakatools::device_buffer<TDev, cms::alpakatools::SimpleVector<SiPixelErrorCompact>>> error_d;
};

#endif  // DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsDevice_h
