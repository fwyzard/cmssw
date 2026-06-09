#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"

#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "Math/RotationZ.h"

#include "Math/Rotation3D.h"
#include "Math/AxisAngle.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"


#include <iterator>    
#include <vector>      

#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp> 

using json = nlohmann::json;

auto clean_p2 = [](double val) {
  return std::abs(val) < 1e-4 ? 0.0 : val;
};

class TrackerGeometryDumper_Phase2 : public edm::one::EDAnalyzer<> {
public:
  explicit TrackerGeometryDumper_Phase2(const edm::ParameterSet& ps);
  ~TrackerGeometryDumper_Phase2() override = default;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopoToken_;
  edm::ESGetToken<GeometricDet, IdealGeometryRecord> geomDetToken_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> ddCompactViewToken_;


  std::string outputFile_;
};

TrackerGeometryDumper_Phase2::TrackerGeometryDumper_Phase2(const edm::ParameterSet& ps)
    : trackerGeomToken_(esConsumes()),
      trackerTopoToken_(esConsumes()),
      geomDetToken_(esConsumes()),
      ddCompactViewToken_(esConsumes<cms::DDCompactView, IdealGeometryRecord>()),
      outputFile_(ps.getUntrackedParameter<std::string>("outputFile", "geometry.json")) {}

void TrackerGeometryDumper_Phase2::analyze(const edm::Event&, const edm::EventSetup& iSetup) {


  const TrackerGeometry& trackerGeom = iSetup.getData(trackerGeomToken_);
  const TrackerTopology& tTopo = iSetup.getData(trackerTopoToken_);

  json surfaces = json::array();

  for (const auto& det : trackerGeom.dets()) {
    DetId detid = det->geographicalId();
    auto surface = det->surface();
    json detJson;

    // DetId
    detJson["geo_id"] = detid.rawId();
    detJson["type"] = "PlaneSurface";

    auto bounds = dynamic_cast<const RectangularPlaneBounds*>(&surface.bounds());
    if (bounds) {
      detJson["bounds"]["type"] = "RectangleBounds";
      detJson["bounds"]["values"] = {   
          -bounds->width() / 2 * 10, // cm to mm
          -bounds->length() / 2 * 10,
          bounds->width() / 2 * 10,
          bounds->length() / 2 * 10
      };
    }
    else{
      auto trap = dynamic_cast<const TrapezoidalPlaneBounds*>(&surface.bounds());
      auto params = trap->parameters(); // half bottom edge, half top edge, half thickness, half apothem
      detJson["bounds"]["type"] = "TrapezoidBounds";
      detJson["bounds"]["values"] = {   
          params[3] * 10, // cm to mm
          params[3] * 10, 
          (params[0] + params[1]) / 2 * 10, 
          0.0
      };

    }

    // Transform (definitions)
    const auto& pos = surface.position();
    const auto& rot = surface.rotation();
    bool notRotated = true;

    // TEST:
    const GeomDet* detPtr = trackerGeom.idToDet(detid);
    const auto subDet = detPtr->subDetector();
    //std::cout << name << std::endl;  // Phase2OTEndcap, Phase2OTBarrel, Phase2PixelEndcap, Phase2PixelBarrel
    // *****

    // TrackerTopology info 
    if (subDet == GeomDetEnumerators::P2PXB) {
      PixelBarrelName pb(detid, &tTopo);
      detJson["topology"] = {
        {"SubDet", "PixelBarrel"},
        {"geo_id", detid.rawId()},
        {"layer", pb.layerName()},
        {"ladder", pb.ladderName()},
        {"module", pb.moduleName()}
      };
      notRotated = true;
    } else if (subDet == GeomDetEnumerators::P2PXEC) {
      PixelEndcapName pe(detid, &tTopo);
      detJson["topology"] = {
        {"SubDet", "PixelEndcap"},
        {"geo_id", detid.rawId()},
        {"disk", pe.diskName()},
        {"blade", pe.bladeName()},
        {"panel", pe.pannelName()},
        {"module", pe.moduleType()}
      };
      notRotated = true;
    }else if (subDet == GeomDetEnumerators::P2OTB) {
      detJson["topology"] = {
        {"SubDet", "OTBarrel"},
        {"geo_id", detid.rawId()},
        {"layer", tTopo.layer(detid)},
        {"side", tTopo.side(detid)},       // 1=backward, 2=forward
        {"string", tTopo.tibString(detid)},
        {"stereo", tTopo.isStereo(detid)}
      };
      notRotated = true;
    }
    else if (subDet == GeomDetEnumerators::P2OTEC) {
      detJson["topology"] = {
        {"SubDet", "OTEndcap"},
        {"geo_id", detid.rawId()},
        {"side", tTopo.side(detid)},
        {"wheel", tTopo.tecWheel(detid)},
        {"ring", tTopo.tecRing(detid)},
        {"petal", tTopo.tecPetalNumber(detid)},
        {"petalSide", tTopo.tecPetalInfo(detid)},  // 1=backward, 2=forward
        {"stereo", tTopo.isStereo(detid)}
      };
	notRotated = true; // No rotation is needed here!
    }
    
    // Load transformation
    detJson["transform"]["translation"] = {pos.x()*10, pos.y()*10, pos.z()*10}; // cm to mm
/*
    // row-major order 
    detJson["transform"]["rotation"] = {
      clean_p2(rot.xx()), clean_p2(rot.xy()), clean_p2(rot.xz()),
      clean_p2(rot.yx()), clean_p2(rot.yy()), clean_p2(rot.yz()),
      clean_p2(rot.zx()), clean_p2(rot.zy()), clean_p2(rot.zz())
    };
*/

    if(notRotated){
      // column-major order 
      detJson["transform"]["rotation"] = {
        clean_p2(rot.xx()), clean_p2(rot.yx()), clean_p2(rot.zx()),  // first column
        clean_p2(rot.xy()), clean_p2(rot.yy()), clean_p2(rot.zy()),  // second column
        clean_p2(rot.xz()), clean_p2(rot.yz()), clean_p2(rot.zz())   // third column
      };
    }
    else{ // For endcap modules, a 90deg rotation (anticlockwise) is applied to the matrix
      // M = rot globale originale
      float M[9] = {
          rot.xx(), rot.yx(), rot.zx(),
          rot.xy(), rot.yy(), rot.zy(),
          rot.xz(), rot.yz(), rot.zz()
      };

      // Rz = rotazione di +90° attorno a Z (nel frame LOCALE del modulo)
      float Rz[9] = {
          0, -1, 0,
          1,  0, 0,
          0,  0, 1
      };

      // M2 = M * Rz  → ruota attorno a Z locale
      float M2[9];
      for (int col = 0; col < 3; ++col) {
        for (int row = 0; row < 3; ++row) {
          M2[row + 3 * col] =
              M[row + 3 * 0] * Rz[0 + 3 * col] +
              M[row + 3 * 1] * Rz[1 + 3 * col] +
              M[row + 3 * 2] * Rz[2 + 3 * col];
        }
      }


      std::cout << "Original X: (" << M[0] << ", " << M[3] << ", " << M[6] << ")\n";
      std::cout << "Rotated  X: (" << clean_p2(M2[0]) << ", " << clean_p2(M2[3]) << ", " << clean_p2(M2[6]) << ")\n";

      std::cout << "Original Y: (" << M[1] << ", " << M[4] << ", " << M[7] << ")\n";
      std::cout << "Rotated  Y: (" << clean_p2(M2[1]) << ", " << clean_p2(M2[4]) << ", " << clean_p2(M2[7]) << ")\n";

      
      detJson["transform"]["rotation"] = {
        clean_p2(M2[0]), clean_p2(M2[1]), clean_p2(M2[2]),  // first column
        clean_p2(M2[3]), clean_p2(M2[4]), clean_p2(M2[5]),  // second column
        clean_p2(M2[6]), clean_p2(M2[7]), clean_p2(M2[8])   // third column
      };
    }


    surfaces.push_back(detJson);
  }

  json output;
  output["surfaces"] = surfaces;

  std::ofstream file(outputFile_);
  file << std::setw(2) << output << std::endl;

  file.close(); 

}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackerGeometryDumper_Phase2);
