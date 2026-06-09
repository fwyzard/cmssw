#pragma once

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Material/ISurfaceMaterial.hpp"
#include "Acts/Material/IVolumeMaterial.hpp"
#include "Acts/Material/TrackingGeometryMaterial.hpp"
#include "Acts/Utilities/EnumBitwiseOperators.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "ActsPlugins/Json/MaterialMapJsonConverter.hpp"

#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace Acts {
class TrackingGeometry;
}  // namespace Acts

enum class JsonFormat : std::uint8_t {
  NoOutput = 0,
  Json = 1,
  Cbor = 2,
  All = std::numeric_limits<std::uint8_t>::max()
};

/// @class Json Material writer
///
/// @brief Writes out Detector material maps
/// using the Json Geometry converter
class JsonMaterialWriter {
 public:
  struct Config {
    /// The config class of the converter
    Acts::MaterialMapJsonConverter::Config converterCfg;
    /// Output file name
    std::string fileName = "material";
    /// Output format of the file
    JsonFormat writeFormat = JsonFormat::Json;
  };

  /// Constructor
  ///
  /// @param config The configuration struct of the writer
  /// @param level The log level
  JsonMaterialWriter(const Config& config, Acts::Logging::Level level);

  /// Virtual destructor
  ~JsonMaterialWriter();

  /// Write out the material map
  ///
  /// @param detMaterial is the SurfaceMaterial and VolumeMaterial maps
  void writeMaterial(
      const Acts::TrackingGeometryMaterial& detMaterial);

  /// Write out the material map from Geometry
  ///
  /// @param tGeometry is the TrackingGeometry
  void write(const Acts::TrackingGeometry& tGeometry);

  /// Readonly access to the config
  const Config& config() const { return m_cfg; }

 private:
  const Acts::Logger& logger() const { return *m_logger; }

  /// The logger instance
  std::unique_ptr<const Acts::Logger> m_logger{nullptr};

  /// The config of the writer
  Config m_cfg;

  /// The material converter
  std::unique_ptr<Acts::MaterialMapJsonConverter> m_converter{nullptr};
};
