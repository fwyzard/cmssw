#ifndef DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsHost_h
#define DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsHost_h

#include <utility>
#include <alpaka/alpaka.hpp>
#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/SimpleVector.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsLayout.h"

class SiPixelDigiErrorsHost : public PortableHostCollection<SiPixelDigiErrorsLayout<>> {
public:
  SiPixelDigiErrorsHost() = default;
  template <typename TQueue>
  explicit SiPixelDigiErrorsHost(int maxFedWords,
                                 cms::alpakatools::host_buffer<SiPixelErrorCompact[]> data,
                                 TQueue queue)
      : PortableHostCollection<SiPixelDigiErrorsLayout<>>(maxFedWords, queue), maxFedWords_(maxFedWords) {
    printf("SiPixelDigiErrorsHost");
    // view().pixelErrors() = std::move(data.data());
    // error_h = cms::alpakatools::make_host_buffer<cms::alpakatools::SimpleVector<SiPixelErrorCompact>>();
    // (*error_h).data()->set_data((*data_h).data());
    view().pixelErrorsVec().set_data(std::move(data.data()));
    printf("ok SiPixelDigiErrorsHost");
  }
  template <typename TQueue>
  explicit SiPixelDigiErrorsHost(int maxFedWords, TQueue queue)
      : PortableHostCollection<SiPixelDigiErrorsLayout<>>(maxFedWords, queue), maxFedWords_(maxFedWords) {
    printf("SiPixelDigiErrorsHost");
    // data_h = cms::alpakatools::make_host_buffer<SiPixelErrorCompact[]>(nErrorWords);
    // error_h = cms::alpakatools::make_host_buffer<cms::alpakatools::SimpleVector<SiPixelErrorCompact>>();
    // (*error_h).data()->set_data((*data_h).data());
    view().pixelErrorsVec().set_data(view().pixelErrors());
    printf("ok SiPixelDigiErrorsHost");
  }
  ~SiPixelDigiErrorsHost() = default;

  SiPixelDigiErrorsHost(SiPixelDigiErrorsHost&&) = default;
  SiPixelDigiErrorsHost& operator=(SiPixelDigiErrorsHost&&) = default;

  int maxFedWords() const { return maxFedWords_; }

  // cms::alpakatools::SimpleVector<SiPixelErrorCompact>* error() { return (*error_h).data(); }
  // cms::alpakatools::SimpleVector<SiPixelErrorCompact> const* error() const { return (*error_h).data(); }
  // auto& error_data() { return (*data_h); }
  // auto const& error_data() const { return (*data_h); }
  // auto& error_vector() const { return (*error_h); }

  cms::alpakatools::SimpleVector<SiPixelErrorCompact>* error() { return (&view().pixelErrorsVec()); }
  cms::alpakatools::SimpleVector<SiPixelErrorCompact> const* error() const { return (&view().pixelErrorsVec()); }
  auto& error_data() { return (*view().pixelErrors()); }
  auto const& error_data() const { return (*view().pixelErrors()); }
  auto& error_vector() const { return view().pixelErrorsVec(); }

private:
  int maxFedWords_ = 0;
  // std::optional<cms::alpakatools::host_buffer<SiPixelErrorCompact[]>> data_h;
  // std::optional<cms::alpakatools::host_buffer<cms::alpakatools::SimpleVector<SiPixelErrorCompact>>> error_h;
};

#endif  // DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsHost_h