#include "ACTSinCMSSW/GeometryDataFormat/interface/CMSDetectorElement.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include "ActsPlugins/Json/AlgebraJsonConverter.hpp"
#include "ActsPlugins/Json/SurfaceJsonConverter.hpp"
#include "Acts/Surfaces/Surface.hpp" 

//ClassImp(Acts::CMSDetectorElement)

namespace Acts {

CMSDetectorElement::CMSDetectorElement(const CMSDetectorElementData& det_data)
  : m_surface(det_data.surf_),
    m_transform(det_data.trans_),
    m_thickness(det_data.thickness_),
    m_detID(det_data.detID_),
    m_subDetector(det_data.subDetector_)
{
    m_surface->assignDetectorElement(*this);
}

const Surface &CMSDetectorElement::surface() const {
  if (!m_surface) {
    throw std::runtime_error("Surface not initialized!");
  }
  return *m_surface;
}

Surface &CMSDetectorElement::surface() {
  if (!m_surface) {
    throw std::runtime_error("Surface not initialized!");
  }
  return *m_surface;
}

const Transform3 &CMSDetectorElement::transform(
    const GeometryContext & /*gctx*/) const {
  return m_transform;
}

const uint32_t &CMSDetectorElement::detID() const {
  return m_detID;
}

const std::string &CMSDetectorElement::subDetector() const {
  return m_subDetector;
}

double CMSDetectorElement::thickness() const {
  return m_thickness;
}

}  // namespace Acts

