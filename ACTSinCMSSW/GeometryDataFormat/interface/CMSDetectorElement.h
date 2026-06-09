#ifndef ACTSinCMSSW_GeometryBuilder_CMSDetectorElement_h
#define ACTSinCMSSW_GeometryBuilder_CMSDetectorElement_h

#include "Acts/Geometry/DetectorElementBase.hpp"

// ROOT includes for serialization
// #include "TObject.h"
// #include "Rtypes.h"

#include <nlohmann/json.hpp>
#include <map>
#include <variant>
#include <string>

namespace Acts {

struct CMSDetectorElementData {
  std::shared_ptr<Surface> surf_ = nullptr;
  double thickness_ = 1;
  Transform3 trans_ = Transform3::Identity();
  uint32_t detID_ = 0;
  std::string subDetector_ = "";
};

class CMSDetectorElement : public DetectorElementBase {
 public:

  CMSDetectorElement(const CMSDetectorElementData& det_data);

  Surface &surface() override;
  const Surface &surface() const override;

  double thickness() const override;

  const Transform3 &transform(const GeometryContext &gctx) const override;

  const uint32_t &detID() const;

  const std::string &subDetector() const;


 private:
  std::shared_ptr<Surface> m_surface = nullptr;
  Transform3 m_transform = Transform3::Identity();
  double m_thickness = 0;
  uint32_t m_detID = 0;
  std::string m_subDetector = "";

};

}  // namespace Acts


#endif
