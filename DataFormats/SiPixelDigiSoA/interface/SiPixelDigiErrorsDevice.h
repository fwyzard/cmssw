#ifndef DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsDevice_h
#define DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsDevice_h

#include <cstdint>
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsHost.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelFormatterErrors.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/SimpleVector.h"

template <typename TDev>
class SiPixelDigiErrorsDevice {
public:
  template <typename TQueue>
  explicit SiPixelDigiErrorsDevice(size_t maxFedWords, SiPixelFormatterErrors errors, TQueue& queue)
      : maxFedWords_(maxFedWords), formatterErrors_h{std::move(errors)} {
    data_d = cms::alpakatools::make_device_buffer<SiPixelErrorCompact[]>(queue, maxFedWords);
    error_d = cms::alpakatools::make_device_buffer<cms::alpakatools::SimpleVector<SiPixelErrorCompact> >(queue);
    (*error_d).data()->construct(maxFedWords, data_d->data());
    ALPAKA_ASSERT_OFFLOAD((*error_d).data()->empty());
    ALPAKA_ASSERT_OFFLOAD((*error_d).data()->capacity() == static_cast<int>(maxFedWords));
  }
  SiPixelDigiErrorsDevice() = default;
  ~SiPixelDigiErrorsDevice() = default;

  SiPixelDigiErrorsDevice(const SiPixelDigiErrorsDevice&) = delete;
  SiPixelDigiErrorsDevice& operator=(const SiPixelDigiErrorsDevice&) = delete;
  SiPixelDigiErrorsDevice(SiPixelDigiErrorsDevice&&) = default;
  SiPixelDigiErrorsDevice& operator=(SiPixelDigiErrorsDevice&&) = default;

  const SiPixelFormatterErrors& formatterErrors() const { return formatterErrors_h; }

  cms::alpakatools::SimpleVector<SiPixelErrorCompact>* error() { return (*error_d).data(); }
  cms::alpakatools::SimpleVector<SiPixelErrorCompact> const* error() const { return (*error_d).data(); }
  cms::alpakatools::SimpleVector<SiPixelErrorCompact> const* c_error() const { return (*error_d).data(); }

  auto& error_vector() const { return (*error_d); }
  auto& error_data() const { return (*data_d); }
  auto maxFedWords() const { return maxFedWords_; }

private:
  int maxFedWords_;
  SiPixelFormatterErrors formatterErrors_h;
  std::optional<cms::alpakatools::device_buffer<TDev, SiPixelErrorCompact[]> > data_d;
  std::optional<cms::alpakatools::device_buffer<TDev, cms::alpakatools::SimpleVector<SiPixelErrorCompact> > > error_d;
};

#endif  // DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsDevice_h
