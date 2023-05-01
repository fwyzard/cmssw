#ifndef DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsHost_h
#define DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsHost_h

#include <utility>

#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelFormatterErrors.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "HeterogeneousCore/AlpakaInterface/interface/SimpleVector.h"

class SiPixelDigiErrorsHost {
public:
  SiPixelDigiErrorsHost() = default;
  ~SiPixelDigiErrorsHost() = default;
  explicit SiPixelDigiErrorsHost(int nErrorWords,
                                 SiPixelFormatterErrors errors,
                                 cms::alpakatools::host_buffer<SiPixelErrorCompact[]> data)
      : nErrorWords_(nErrorWords), formatterErrors_h{std::move(errors)} {
    data_h = std::move(data);
    error_h = cms::alpakatools::make_host_buffer<cms::alpakatools::SimpleVector<SiPixelErrorCompact>>();
    (*error_h).data()->set_data((*data_h).data());
  }
  explicit SiPixelDigiErrorsHost(int nErrorWords, SiPixelFormatterErrors errors)
      : nErrorWords_(nErrorWords), formatterErrors_h{std::move(errors)} {
    data_h = cms::alpakatools::make_host_buffer<SiPixelErrorCompact[]>(nErrorWords);
    error_h = cms::alpakatools::make_host_buffer<cms::alpakatools::SimpleVector<SiPixelErrorCompact>>();
    (*error_h).data()->set_data((*data_h).data());
  }

  int nErrorWords() const { return nErrorWords_; }

  cms::alpakatools::SimpleVector<SiPixelErrorCompact>* error() { return (*error_h).data(); }
  cms::alpakatools::SimpleVector<SiPixelErrorCompact> const* error() const { return (*error_h).data(); }
  auto& error_data() { return (*data_h); }
  auto const& error_data() const { return (*data_h); }
  auto& error_vector() const { return (*error_h); }

  const SiPixelFormatterErrors& formatterErrors() const { return formatterErrors_h; }

private:
  int nErrorWords_ = 0;
  SiPixelFormatterErrors formatterErrors_h;
  std::optional<cms::alpakatools::host_buffer<SiPixelErrorCompact[]>> data_h;
  std::optional<cms::alpakatools::host_buffer<cms::alpakatools::SimpleVector<SiPixelErrorCompact>>> error_h;
};

#endif  // DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsHost_h