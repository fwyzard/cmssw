#ifndef DataFormats_SiPixelDigi_SiPixelDigisHost_h
#define DataFormats_SiPixelDigi_SiPixelDigisHost_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/Portable/interface/PortableHostCollection.h"

#include "SiPixelDigisLayout.h"

// TODO: The class is created via inheritance of the PortableDeviceCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306
class SiPixelDigisHost : public PortableHostCollection<SiPixelDigisLayout<>> {
public:
  SiPixelDigisHost() = default;
  template <typename TQueue>
  explicit SiPixelDigisHost(size_t maxFedWords, TQueue queue)
      : PortableHostCollection<SiPixelDigisLayout<>>(maxFedWords + 1, queue) {}
  ~SiPixelDigisHost() = default;

  SiPixelDigisHost(SiPixelDigisHost &&) = default;
  SiPixelDigisHost &operator=(SiPixelDigisHost &&) = default;

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

#endif  // DataFormats_SiPixelDigi_SiPixelDigisHost_h
