// TrackerGeometryDumper.h
#ifndef TrackerGeometryDumper_h
#define TrackerGeometryDumper_h

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include <fstream>
#include <nlohmann/json.hpp> // puoi aggiungere nlohmann_json nelle build

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Geometry/Polyhedron.hpp"
#include "Acts/Surfaces/PlaneSurface.hpp"
#include "Acts/Surfaces/RectangleBounds.hpp"

class TrackerGeometryDumper : public edm::one::EDAnalyzer<> {
public:
  explicit TrackerGeometryDumper(const edm::ParameterSet& p);
  ~TrackerGeometryDumper() override = default;

  void analyze(const edm::Event& e, const edm::EventSetup& es) override;

private:
  void writeDetInfo(const GeomDetUnit* det, nlohmann::json& j);
  nlohmann::json detIdInfo(DetId id);

  std::string outFileName_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopoToken_;
  std::ofstream outFile_;
};

#endif

