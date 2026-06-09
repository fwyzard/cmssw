#include "ACTSinCMSSW/GeometryDataFormat/interface/JsonMaterialWriter.hpp"

#include "Acts/Utilities/Helpers.hpp"

#include <fstream>
#include <iomanip>
#include <ios>
#include <vector>

#include <nlohmann/json.hpp>

JsonMaterialWriter::JsonMaterialWriter(
    const JsonMaterialWriter::Config& config,
    Acts::Logging::Level level)
    : m_logger{Acts::getDefaultLogger("JsonMaterialWriter", level)},
      m_cfg(config),
      m_converter{std::make_unique<Acts::MaterialMapJsonConverter>(
          m_cfg.converterCfg, level)} {}

JsonMaterialWriter::~JsonMaterialWriter() = default;

void JsonMaterialWriter::writeMaterial(
    const Acts::TrackingGeometryMaterial& detMaterial) {
  // Evoke the converter
  auto jOut = m_converter->materialMapsToJson(detMaterial);
  // And write the file(s)

  std::string fileName = m_cfg.fileName;
  ACTS_VERBOSE("Writing to file: " << fileName);
  std::ofstream ofj(fileName);
  ofj << std::setw(4) << jOut << std::endl;
  
}

void JsonMaterialWriter::write(
    const Acts::TrackingGeometry& tGeometry) {
  // Evoke the converter
  auto jOut = m_converter->trackingGeometryToJson(tGeometry);
  // And write the file(s)
  std::ofstream ofj(m_cfg.fileName + ".json");
  ofj << std::setw(4) << jOut << std::endl;
  
}