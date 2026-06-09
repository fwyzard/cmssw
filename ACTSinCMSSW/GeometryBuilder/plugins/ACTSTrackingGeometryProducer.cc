/// ######################################################################################
/// # Plugin used to convert the CMSSW detector elements in ACTS detector elements.      #
/// # The output is a vector of Acts::CMSDetectorElement which can be used to            #
/// # build the Tracking Geometry.                                                       #
/// # NOTE: Plugins list:                                                                #  
/// # (1) Converts the CMSSW detElements into ACTS ones and builds the Tracking Geometry # <- DONE (I) <->(III) flags to map material
/// # (2) Produces a JSON material file starting from material tracks                    # <- DONE (II)
/// # (3) Takes as input the tracking geometry from (1) and the json file from (3) and   # <- DONE (IV) validation only
/// #     decorates the tracking geometry with material. It performs material Validation #  
/// ######################################################################################
// git cms-addpkg
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Utilities/interface/typelookup.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/ACTSTrackerGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"

#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "Math/RotationZ.h"
#include "FWCore/Framework/interface/ESConsumesCollector.h"

#include "Math/Rotation3D.h"
#include "Math/AxisAngle.h"

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Detector/KdtSurfacesProvider.hpp"
#include "Acts/Geometry/Polyhedron.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Geometry/TrackingGeometryVisitor.hpp"
#include "Acts/Surfaces/PlaneSurface.hpp"
#include "Acts/Surfaces/RectangleBounds.hpp"
#include "Acts/Surfaces/TrapezoidBounds.hpp"
#include "Acts/Visualization/GeometryView3D.hpp"
#include "Acts/Visualization/ObjVisualization3D.hpp"
#include "ActsPlugins/Json/JsonDetectorElement.hpp"
#include "Acts/Visualization/ViewConfig.hpp"

#include "Acts/Geometry/Blueprint.hpp"
#include "Acts/Geometry/ContainerBlueprintNode.hpp" 
#include "Acts/Geometry/LayerBlueprintNode.hpp" 
#include "Acts/Geometry/MaterialDesignatorBlueprintNode.hpp" 
#include "Acts/Geometry/SurfaceArrayCreator.hpp"
#include "Acts/Detector/KdtSurfacesProvider.hpp"
#include "Acts/Detector/ProtoDetector.hpp"
#include "Acts/Detector/detail/ReferenceGenerators.hpp"
#include "Acts/Detector/interface/ISurfacesProvider.hpp"
#include "Acts/Detector/LayerStructureBuilder.hpp"
#include "Acts/Detector/detail/BlueprintHelper.hpp"
#include "Acts/Detector/detail/BlueprintDrawer.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/Navigation/SurfaceArrayNavigationPolicy.hpp"
#include "Acts/Navigation/TryAllNavigationPolicy.hpp"
#include "ActsPlugins/ActSVG/DetectorSvgConverter.hpp"
#include "ActsPlugins/Json/JsonSurfacesReader.hpp"
#include "ActsPlugins/Json/JsonMaterialDecorator.hpp"
#include "ActsPlugins/Json/SurfaceJsonConverter.hpp"
#include "ActsPlugins/ActSVG/TrackingGeometrySvgConverter.hpp"
#include "ActsPlugins/ActSVG/SurfaceArraySvgConverter.hpp"
#include "ActsPlugins/Root/RootMaterialTrackIo.hpp"

#include "Acts/Material/PropagatorMaterialAssigner.hpp"
#include "Acts/Material/BinnedSurfaceMaterialAccumulater.hpp"
#include "Acts/Material/BinnedSurfaceMaterial.hpp"
#include "Acts/Material/AccumulatedMaterialSlab.hpp"
#include "Acts/Material/HomogeneousSurfaceMaterial.hpp"
#include "Acts/Material/IntersectionMaterialAssigner.hpp"
#include "Acts/Material/MaterialValidater.hpp"
#include "Acts/Material/MaterialMapper.hpp"

#include <fstream>
#include <iomanip>
#include <random>
#include <nlohmann/json.hpp> 
#include "TChain.h"
#include "TFile.h"

#include <iostream>
#include <string>
#include <vector>

#include "ACTSinCMSSW/GeometryDataFormat/interface/CMSDetectorElement.h"
#include "ACTSinCMSSW/GeometryDataFormat/interface/JsonMaterialWriter.hpp"

using json = nlohmann::json;
using KdtSurfacesDim2Bin100 = Acts::Experimental::KdtSurfaces<2u, 100u>;
const std::array<Acts::AxisDirection, 2UL> casts{Acts::AxisDirection::AxisZ, Acts::AxisDirection::AxisR};

using DetElVect = std::vector<std::shared_ptr<Acts::CMSDetectorElement>>;

TYPELOOKUP_DATA_REG(Alignments);

// DEBUG: function to identify backsided modules:

// Robust inner/outer classification for tilted sensors:
// Use sign of normal x position (position from origin).
// If negative, the plane normal points toward the beamline => "inner-facing".
int innerOuterFromOrientation(const TrackerTopology & tTopo, DetId detId, const GlobalPoint& pos, const GlobalVector& nrm) {
// const double innerOuterFromOrientation(const TrackerTopology & tTopo, DetId detId, const GlobalPoint& pos, const GlobalVector& nrm) {
  const double dot = pos.x() * nrm.x() + pos.y() * nrm.y() + pos.z() * nrm.z();
  const bool normalOut = (dot > 0.0);
  // Which member of the stack is this?
  const bool isLower = tTopo.isLower(detId);  // <-- this is the key topo query

  // Map to inner(0)/outer(1)
  // If normal points outward, lower sensor is inner and upper is outer.
  // If normal points inward, mapping flips.
  const int innerOuter = normalOut ? (isLower ? 0 : 1) : (isLower ? 1 : 0);
  return innerOuter;
  // return normalOut;
}

// END of debug  for identify backsided modules


struct TrackingGeometryWithDetEls {
    DetElVect detElements;
    std::shared_ptr<Acts::TrackingGeometry> trackingGeometry;
};
TYPELOOKUP_DATA_REG(TrackingGeometryWithDetEls);

// ############################## MATERIAL METHODS ##############################

struct MaterialConfig {
  /// material collection to read
  std::string outputMaterialTracks = "material-tracks";
  /// name of the output tree
  std::string treeName = "material-tracks";
  /// List of input files
  std::vector<std::string> fileList;

  // Read surface information for the root file
  bool readCachedSurfaceInformation = false;
};

template <typename element_t, typename index_t>
void stableSort(index_t numElements, const element_t* elements,
                index_t* sortedIndices, Bool_t sortDescending) {
  for (index_t i = 0; i < numElements; i++) {
    sortedIndices[i] = i;
  }

  if (sortDescending) {
    std::stable_sort(sortedIndices, sortedIndices + numElements,
                     CompareDesc<const element_t*>(elements));
  } else {
    std::stable_sort(sortedIndices, sortedIndices + numElements,
                     CompareAsc<const element_t*>(elements));
  }
}

class MyMaterialEvReader {
public:
    MyMaterialEvReader(const MaterialConfig& config)
      : m_cfg(config){

      ActsPlugins::RootMaterialTrackIo::Config ioCfg;
      m_payload = std::make_unique<ActsPlugins::RootMaterialTrackIo>(ioCfg);
      
      m_inputChain = std::make_unique<TChain>(m_cfg.treeName.c_str());

      // loop over the input files
      for (const auto& inputFile : m_cfg.fileList) {
        // add file to the input chain
        m_inputChain->Add(inputFile.c_str());
        //std::cout << "Adding File " << inputFile << " to tree '" << m_cfg.treeName << "'." << std::endl;
      }

      // Connect the branches
      m_payload->connectForRead(*m_inputChain);

      // get the number of entries, which also loads the tree
      std::size_t nentries = m_inputChain->GetEntries();
      m_events = static_cast<std::size_t>(m_inputChain->GetMaximum("event_id") + 1);
      m_batchSize = nentries / m_events;
      //std::cout << "The full chain has "
      //           << nentries << " entries for " << m_events
      //           << " events this corresponds to a batch size of: " << m_batchSize << std::endl;

      // Sort the entry numbers of the events
      {
        // necessary to guarantee that m_inputChain->GetV1() is valid for the
        // entire range
        m_inputChain->SetEstimate(nentries + 1);

        m_entryNumbers.resize(nentries);
        m_inputChain->Draw("event_id", "", "goff");
        stableSort(m_inputChain->GetEntries(), m_inputChain->GetV1(),
                                m_entryNumbers.data(), false);
      }
    }

    std::unordered_map<std::size_t, Acts::RecordedMaterialTrack> read(std::size_t eventNumber) {

      mtrackCollection.clear();
      
      // Loop over the entries for this event
      for (std::size_t ib = 0; ib < m_batchSize; ++ib) {
        // Read the correct entry: startEntry + ib
        auto entry = m_batchSize * eventNumber + ib;
        entry = m_entryNumbers.at(entry);
        //std::cout << "Reading event: " << eventNumber
        //                               << " with stored entry: " << entry << std::endl; 
        m_inputChain->GetEntry(entry);

        Acts::RecordedMaterialTrack rmTrack = m_payload->read();

        //std::cout << "Track vertex:  " << rmTrack.first.first << std::endl;
        //std::cout << "Track momentum:" << rmTrack.first.second << std::endl;

        mtrackCollection[ib] = (std::move(rmTrack));
      }

      return mtrackCollection;
    }

private:
    MaterialConfig m_cfg;
    std::unique_ptr<TChain> m_inputChain;
    std::unique_ptr<ActsPlugins::RootMaterialTrackIo> m_payload;
    
    std::vector<long long> m_entryNumbers = {};

    std::size_t m_events = 0;
    std::size_t m_batchSize = 0;

    std::unordered_map<std::size_t, Acts::RecordedMaterialTrack> mtrackCollection;
};

struct myContext{
  /// Magnetic and Geometry contrext
  Acts::GeometryContext geoContext;
  Acts::MagneticFieldContext magFieldContext;
  /// Number of event
  std::size_t EvNumber; 
};

// ===== Helper function to visualize the sufaces =====
void writeSurfacesObj(const std::vector<std::shared_ptr<Acts::Surface>>& surfaces, 
                      const std::string& fileName)
{
    Acts::ObjVisualization3D obj;

    for (size_t i = 0; i < surfaces.size(); ++i) {
        if (!surfaces[i]) {
            std::cerr << "[ERROR] Surface " << i << " is nullptr!" << std::endl;
            continue;
        }

        Acts::GeometryView3D::drawSurface(
            obj,
            *surfaces[i],
            Acts::GeometryContext{},
            Acts::Transform3::Identity(),
            Acts::ViewConfig{}
        );
    }

    obj.write(fileName);
    obj.clear();
};

Acts::Transform3 GenerateTranslation(double dx, double dy, double dz) {
    return Acts::Transform3::Identity() * Acts::Translation3{Acts::Vector3{dx, dy, dz}};
};

std::vector<std::shared_ptr<Acts::Surface>> SelectActiveSurfaces_PhaseI(const KdtSurfacesDim2Bin100& surfaces, std::string Layer_name){

  Acts::RangeXD<2, double, std::array> Range({0, 0}, {0, 0});

  if(Layer_name=="PixL0"){
    Range = Acts::RangeXD<2, double, std::array>({-290, 15}, {280, 35});
  }
  else if(Layer_name=="PixL1"){
    Range = Acts::RangeXD<2, double, std::array>({-290,40}, {280,80});
  }
  else if(Layer_name=="PixL2"){
    Range = Acts::RangeXD<2, double, std::array>({-290,85}, {280,150});
  }
  else if(Layer_name=="PixL3"){
    Range = Acts::RangeXD<2, double, std::array>({-290,150}, {280,200});
  }
  else if(Layer_name=="PixelPos2"){
    Range = Acts::RangeXD<2, double, std::array>({450,25}, {1160,220}); //550
  }
  else if(Layer_name=="PixelPos1"){
    Range = Acts::RangeXD<2, double, std::array>({350,25}, {445,220});
  }
  else if(Layer_name=="PixelPos0"){
    Range = Acts::RangeXD<2, double, std::array>({285,25}, {345,220});
  }
  else if(Layer_name=="PixelNeg2"){
    Range = Acts::RangeXD<2, double, std::array>({-1160,25}, {-450,220});
  }
  else if(Layer_name=="PixelNeg1"){
    Range = Acts::RangeXD<2, double, std::array>({-445,25}, {-350,220});
  }
  else if(Layer_name=="PixelNeg0"){
    Range = Acts::RangeXD<2, double, std::array>({-345,25}, {-285,220});
  }
  else if(Layer_name=="TIDNeg2"){
    Range = Acts::RangeXD<2, double, std::array>({-1160,220}, {-1000,530});
  }
  else if(Layer_name=="TIDNeg1"){
    Range = Acts::RangeXD<2, double, std::array>({-1000,220}, {-850,530});
  }
  else if(Layer_name=="TIDNeg0"){
    Range = Acts::RangeXD<2, double, std::array>({-850,220}, {-700,530});
  }
  else if(Layer_name=="TIB3"){
    Range = Acts::RangeXD<2, double, std::array>({-700,480}, {700,560});
  }
  else if(Layer_name=="TIB2"){
    Range = Acts::RangeXD<2, double, std::array>({-700,400}, {700,480});
  }
  else if(Layer_name=="TIB1"){
    Range = Acts::RangeXD<2, double, std::array>({-700,300}, {700,400});
  }
  else if(Layer_name=="TIB0"){
    Range = Acts::RangeXD<2, double, std::array>({-700,200}, {700,300});
  }
  else if(Layer_name=="TIDPos2"){
    Range = Acts::RangeXD<2, double, std::array>({1000,220}, {1160,530});
  }
  else if(Layer_name=="TIDPos1"){
    Range = Acts::RangeXD<2, double, std::array>({850,220}, {1000,530});
  }
  else if(Layer_name=="TIDPos0"){
    Range = Acts::RangeXD<2, double, std::array>({700,220}, {850,530});
  }
  else if(Layer_name=="NegTEC8"){
    Range = Acts::RangeXD<2, double, std::array>({  -2800,220}, {   -2550,1200});
  }
  else if(Layer_name=="NegTEC7"){
    Range = Acts::RangeXD<2, double, std::array>({  -2550,220}, {   -2350,1200});
  }
  else if(Layer_name=="NegTEC6"){
    Range = Acts::RangeXD<2, double, std::array>({  -2350,220}, {   -2150,1200});
  }
  else if(Layer_name=="NegTEC5"){
    Range = Acts::RangeXD<2, double, std::array>({  -2150,220}, {   -1950,1200});
  }
  else if(Layer_name=="NegTEC4"){
    Range = Acts::RangeXD<2, double, std::array>({  -1950,220}, {   -1810,1200});
  }
  else if(Layer_name=="NegTEC3"){
    Range = Acts::RangeXD<2, double, std::array>({  -1810,220}, {   -1675,1200});
  }
  else if(Layer_name=="NegTEC2"){
    Range = Acts::RangeXD<2, double, std::array>({  -1675,220}, {   -1525,1200});
  }
  else if(Layer_name=="NegTEC1"){
    Range = Acts::RangeXD<2, double, std::array>({  -1525,220}, {   -1400,1200});
  }
  else if(Layer_name=="NegTEC0"){
    Range = Acts::RangeXD<2, double, std::array>({  -1400,220}, {   -1200,1200});
  }
  else if(Layer_name=="TOB0"){
    Range = Acts::RangeXD<2, double, std::array>({  -1160,560}, {    1160, 650});
  }
  else if(Layer_name=="TOB1"){
    Range = Acts::RangeXD<2, double, std::array>({-1160, 650}, {1160, 750});
  }
  else if(Layer_name=="TOB2"){
    Range = Acts::RangeXD<2, double, std::array>({-1160, 750}, {1160, 840});
  }
  else if(Layer_name=="TOB3"){
    Range = Acts::RangeXD<2, double, std::array>({-1160, 840}, {1160, 940});
  }
  else if(Layer_name=="TOB4"){
    Range = Acts::RangeXD<2, double, std::array>({-1160, 940}, {1160, 1040});
  }
  else if(Layer_name=="TOB5"){
    Range = Acts::RangeXD<2, double, std::array>({-1160, 1040}, {1160, 1200});
  }
  else if(Layer_name=="PosTEC8"){
    Range = Acts::RangeXD<2, double, std::array>({  2550, 220}, {   2800,1200});
  }
  else if(Layer_name=="PosTEC7"){
    Range = Acts::RangeXD<2, double, std::array>({  2350, 220}, {   2550,1200});
  }
  else if(Layer_name=="PosTEC6"){
    Range = Acts::RangeXD<2, double, std::array>({  2150, 220}, {   2350,1200});
  }
  else if(Layer_name=="PosTEC5"){
    Range = Acts::RangeXD<2, double, std::array>({  1950, 220}, {   2150,1200});
  }
  else if(Layer_name=="PosTEC4"){
    Range = Acts::RangeXD<2, double, std::array>({  1810, 220}, {   1950,1200});
  }
  else if(Layer_name=="PosTEC3"){
    Range = Acts::RangeXD<2, double, std::array>({  1675, 220}, {   1810,1200});
  }
  else if(Layer_name=="PosTEC2"){
    Range = Acts::RangeXD<2, double, std::array>({  1525, 220}, {   1675,1200});
  }
  else if(Layer_name=="PosTEC1"){
    Range = Acts::RangeXD<2, double, std::array>({  1400, 220}, {   1525,1200});
  }
  else if(Layer_name=="PosTEC0"){
    Range = Acts::RangeXD<2, double, std::array>({  1200, 220}, {   1400,1200});
  }

  std::vector<std::shared_ptr<Acts::Surface>> ActiveSurfaces = surfaces.surfaces(Range);

  return ActiveSurfaces;
}

void makeBinning(auto& layer, Acts::SurfaceArrayNavigationPolicy::LayerType layer_type, double Bin0, double Bin1){
  layer.setNavigationPolicyFactory(Acts::NavigationPolicyFactory()
                            .add<Acts::SurfaceArrayNavigationPolicy>(
                                Acts::SurfaceArrayNavigationPolicy::Config{
                                    .layerType = layer_type,
                                    .bins = {Bin0, Bin1}}) 
                            .add<Acts::TryAllNavigationPolicy>(
                                Acts::TryAllNavigationPolicy::Config{.sensitives = true})                                          
                            .asUniquePtr()); 
}

void AddExtraLayer(std::string gapName, 
                   bool isBarrel, 
                   Acts::Experimental::CylinderContainerBlueprintNode* cont,
                   Acts::Transform3 transform,
                   std::shared_ptr<Acts::CylinderVolumeBounds> bounds){

  Acts::Transform3 base{Acts::Transform3::Identity()};  
  Acts::AxisDirection DirectionBuild;
  if(!isBarrel){
    DirectionBuild =  Acts::AxisDirection::AxisZ;
  } else {
    DirectionBuild =  Acts::AxisDirection::AxisR;
  }              
  cont->addCylinderContainer(gapName.c_str(), DirectionBuild, [&](auto& gap) {
    gap.setAttachmentStrategy(Acts::VolumeAttachmentStrategy::Gap)
       .setResizeStrategy(Acts::VolumeResizeStrategy::Gap);

    gap.addMaterial(gapName.c_str(), [&](Acts::Experimental::MaterialDesignatorBlueprintNode& mat){
      if(!isBarrel){
        mat.configureFace(Acts::CylinderVolumeBounds::Face::NegativeDisc, {Acts::AxisDirection::AxisR, Acts::AxisBoundaryType::Bound, 100}, {Acts::AxisDirection::AxisPhi, Acts::AxisBoundaryType::Bound, 100});
        mat.configureFace(Acts::CylinderVolumeBounds::Face::PositiveDisc, {Acts::AxisDirection::AxisR, Acts::AxisBoundaryType::Bound, 100}, {Acts::AxisDirection::AxisPhi, Acts::AxisBoundaryType::Bound, 100});
      }
      else{
        mat.configureFace(Acts::CylinderVolumeBounds::Face::OuterCylinder, {Acts::AxisDirection::AxisRPhi, Acts::AxisBoundaryType::Bound, 100}, {Acts::AxisDirection::AxisZ, Acts::AxisBoundaryType::Bound, 100});                  
      }

      mat.addCylinderContainer(gapName.c_str(), Acts::AxisDirection::AxisZ, [&](auto& L) {
        L.addStaticVolume(base * transform, bounds, gapName.c_str());
      });
    });       
  });

};

struct MatSurfaceSelector : public Acts::TrackingGeometryMutableVisitor {
  std::vector<Acts::Surface*> surfaces;
  void visitSurface(Acts::Surface& surface) override {
    if(surface.surfaceMaterial() != nullptr && !rangeContainsValue(surfaces, &surface)) {
        surfaces.push_back(&surface);
    }
  }
};

// struct MaterialSurfaceSelector {
//     std::vector<const Acts::Surface*> surfaces = {};

//     /// @param surface is the test surface
//     void operator()(const Acts::Surface* surface) {
//       if (surface->surfaceMaterial() != nullptr && !rangeContainsValue(surfaces, surface)) {
//         // surfaces.push_back(surface);
//         surfaces.push_back(const_cast<Acts::Surface*>(surface));
//       }
//     }
// };

template <typename MakeLayerFn>
void AddDiskLayer_and_Material(Acts::Experimental::CylinderContainerBlueprintNode* cont,
                               std::string LayerName,
                               MakeLayerFn&& makeLayerFunc,
                               double z_pos,
                               double bin0,
                               double bin1){
  
    std::string matName = LayerName + "_Material";
    Acts::Transform3 base{Acts::Transform3::Identity()};

    cont->addMaterial(matName.c_str(), [&](Acts::Experimental::MaterialDesignatorBlueprintNode& mat) {
      //mat.configureFace(Acts::CylinderVolumeBounds::Face::OuterCylinder, {Acts::AxisDirection::AxisRPhi, Acts::AxisBoundaryType::Bound, 20}, {Acts::AxisDirection::AxisZ, Acts::AxisBoundaryType::Bound, 20});
      mat.configureFace(Acts::CylinderVolumeBounds::Face::NegativeDisc, {Acts::AxisDirection::AxisR, Acts::AxisBoundaryType::Bound, 100}, {Acts::AxisDirection::AxisPhi, Acts::AxisBoundaryType::Bound, 100});
      mat.configureFace(Acts::CylinderVolumeBounds::Face::PositiveDisc, {Acts::AxisDirection::AxisR, Acts::AxisBoundaryType::Bound, 100}, {Acts::AxisDirection::AxisPhi, Acts::AxisBoundaryType::Bound, 100});
      //mat.configureFace(Acts::CylinderVolumeBounds::Face::InnerCylinder, {Acts::AxisDirection::AxisRPhi, Acts::AxisBoundaryType::Bound, 20}, {Acts::AxisDirection::AxisZ, Acts::AxisBoundaryType::Bound, 20});
      

      mat.addCylinderContainer(LayerName, Acts::AxisDirection::AxisZ, [&](auto& L) {
        L.addLayer(LayerName,[&](auto& layer) {
                      makeLayerFunc(base * Acts::Translation3{Acts::Vector3{0, 0, z_pos}}, layer, LayerName, bin0, bin1);
                  });
      });
    });
};

void AddCylinderLayer_and_Material(Acts::Experimental::CylinderContainerBlueprintNode* cont,
                                  std::string LayerName,
                                  KdtSurfacesDim2Bin100& KdtSurfaces, 
                                  double bin0,
                                  double bin1){

  std::string matName = LayerName + "_Material";
  Acts::Transform3 base{Acts::Transform3::Identity()};

  cont->addMaterial(matName.c_str(), [&](Acts::Experimental::MaterialDesignatorBlueprintNode& mat) {
                    mat.configureFace(Acts::CylinderVolumeBounds::Face::OuterCylinder, {Acts::AxisDirection::AxisRPhi, Acts::AxisBoundaryType::Bound, 100}, {Acts::AxisDirection::AxisZ, Acts::AxisBoundaryType::Bound, 100});
                    //mat.configureFace(Acts::CylinderVolumeBounds::Face::NegativeDisc, {Acts::AxisDirection::AxisR, Acts::AxisBoundaryType::Bound, 15}, {Acts::AxisDirection::AxisPhi, Acts::AxisBoundaryType::Bound, 25});
                    //mat.configureFace(Acts::CylinderVolumeBounds::Face::PositiveDisc, {Acts::AxisDirection::AxisR, Acts::AxisBoundaryType::Bound, 15}, {Acts::AxisDirection::AxisPhi, Acts::AxisBoundaryType::Bound, 25});
                    //mat.configureFace(Acts::CylinderVolumeBounds::Face::InnerCylinder, {Acts::AxisDirection::AxisRPhi, Acts::AxisBoundaryType::Bound, 20}, {Acts::AxisDirection::AxisZ, Acts::AxisBoundaryType::Bound, 20});
                    
                    mat.addCylinderContainer(LayerName, Acts::AxisDirection::AxisR, [&](auto& L) {
                      L.addLayer(LayerName, [&](auto& layer) {
                        std::vector<std::shared_ptr<Acts::Surface>> surfaces = SelectActiveSurfaces_PhaseI(KdtSurfaces, LayerName);
                        // Binning:
                        makeBinning(layer, Acts::SurfaceArrayNavigationPolicy::LayerType::Cylinder, bin0, bin1);

                        if(LayerName.find("TOB") == std::string::npos){
                          layer.setSurfaces(surfaces)
                              .setLayerType(Acts::Experimental::LayerBlueprintNode::LayerType::Cylinder)
                              .setEnvelope(Acts::ExtentEnvelope{{
                                  .z = {5*Acts::UnitConstants::mm, 5*Acts::UnitConstants::mm},
                                  .r = {1*Acts::UnitConstants::mm, 1*Acts::UnitConstants::mm},
                              }})
                              .setTransform(base);
                        }
                        else{
                          layer.setSurfaces(surfaces)
                              .setLayerType(Acts::Experimental::LayerBlueprintNode::LayerType::Cylinder)
                              .setEnvelope(Acts::ExtentEnvelope{{
                                  .z = {5*Acts::UnitConstants::mm, 5*Acts::UnitConstants::mm},
                                  .r = {1*Acts::UnitConstants::mm, 1*Acts::UnitConstants::mm},
                              }})
                              .setTransform(base)
                              .setUseCenterOfGravity(false, false, false);
                        }
                            
                      })
                      .setUseCenterOfGravity(false, false, false);
                    });
                  });
};

// ##################### Decorator Methods #################################################                     


struct MaterialValidation_cfg{
  /// Number of tracks per event
  std::size_t ntracks = 1000;
  /// Start position for the scan
  Acts::Vector3 startPosition = Acts::Vector3(0., 0., 0.);
  /// Start direction for the scan: phi
  std::pair<double, double> phiRange = {-std::numbers::pi, std::numbers::pi};
  /// Start direction for the scan: eta
  std::pair<double, double> etaRange = {-4., 4.};
  // The validater
  std::shared_ptr<Acts::MaterialValidater> materialValidater = nullptr;
  /// Output collection name
  std::string outputMaterialTracks = "material_tracks";

};

using RandomEngine = std::mt19937;
using RandomSeed = uint32_t;


class MaterialValidator {
public:
  MaterialValidator(const MaterialValidation_cfg& cfg)
  : m_cfg(cfg){
    if(m_cfg.materialValidater == nullptr){
      throw std::invalid_argument("Missing material validater.");
    }
  }

  std::unordered_map<std::size_t, Acts::RecordedMaterialTrack> execute(const myContext& context, bool debug = false){
    // Create a random number generator
    RandomSeed seed = 1234567890u;
    RandomEngine rng(seed + context.EvNumber);

    // Setup random number distributions for some quantities
    std::uniform_real_distribution<double> phiDist(m_cfg.phiRange.first, m_cfg.phiRange.second);
    std::uniform_real_distribution<double> etaDist(m_cfg.etaRange.first, m_cfg.etaRange.second);

    // Loop over the number of tracks
    for (std::size_t iTrack = 0; iTrack < m_cfg.ntracks; ++iTrack) {
      if(debug) std::cout << "\rEv: " << context.EvNumber << ", Track " << iTrack + 1 << " out of " << m_cfg.ntracks << std::flush;
      // Generate a random phi and eta
      double phi = phiDist(rng);
      double eta = etaDist(rng);
      double theta = 2 * std::atan(std::exp(-eta));
      Acts::Vector3 direction(std::cos(phi) * std::sin(theta), std::sin(phi) * std::sin(theta), std::cos(theta));

      // Record the material
      auto rMaterial = m_cfg.materialValidater->recordMaterial(
          context.geoContext, context.magFieldContext, m_cfg.startPosition,
          direction);

      recordedMaterialTracks[iTrack] = rMaterial;
    }

    return recordedMaterialTracks;
  }

private:
  MaterialValidation_cfg m_cfg;
  std::unordered_map<std::size_t, Acts::RecordedMaterialTrack> recordedMaterialTracks;

};

struct MyMaterialTrackWriter_cfg {
  /// material collection to write
  std::string inputMaterialTracks = "material-tracks";
  /// path of the output file
  std::string filePath = "";
  /// file access mode
  std::string fileMode = "RECREATE";
  /// name of the output tree
  std::string treeName = "material-tracks";

  /// Re-calculate total values from individual steps (for cross-checks)
  bool recalculateTotals = false;
  /// Write aut pre and post step (for G4), otherwise central step position
  bool prePostStep = false;
  /// Write the surface to which the material step correpond
  bool storeSurface = false;
  /// Write the volume to which the material step correpond
  bool storeVolume = false;
  /// Collapse consecutive interactions of a single surface into a single
  /// interaction
  bool collapseInteractions = false;
};

class MyMaterialTrackWriter {
 public:
      MyMaterialTrackWriter(const MyMaterialTrackWriter_cfg& config)
        : m_cfg(config){

      ActsPlugins::RootMaterialTrackIo::Config ioCfg;
      m_payload = std::make_unique<ActsPlugins::RootMaterialTrackIo>(ioCfg);

      // An input collection name and tree name must be specified
      if (m_cfg.inputMaterialTracks.empty()) {
        throw std::invalid_argument("Missing input collection");
      } else if (m_cfg.treeName.empty()) {
        throw std::invalid_argument("Missing tree name");
      }

      // Setup ROOT I/O
      m_outputFile = TFile::Open(m_cfg.filePath.c_str(), m_cfg.fileMode.c_str());
      if (m_outputFile == nullptr) {
        throw std::ios_base::failure("Could not open '" + m_cfg.filePath + "'");
      }

      m_outputFile->cd();
      m_outputTree = new TTree(m_cfg.treeName.c_str(), "TTree from RootMaterialTrackWriter");
      if (m_outputTree == nullptr) {
        throw std::bad_alloc();
      }
      // Connect the branches
      m_payload->connectForWrite(*m_outputTree);
      }

      ~MyMaterialTrackWriter() {
          m_outputTree = nullptr;
          m_outputFile = nullptr;
      }

      void finalize() {
        if (m_outputFile && m_outputTree) {
            m_outputFile->cd();
            m_outputTree->Write();
            m_outputFile->Close();
        }
        m_outputTree = nullptr;
        m_outputFile = nullptr;
      }

      void writeT(
          const myContext& ctx, 
          const std::unordered_map<std::size_t, Acts::RecordedMaterialTrack>& materialTracks) {

        // Loop over the material tracks and write them out
        for (auto& [idTrack, mtrack] : materialTracks) {
          m_payload->write(ctx.geoContext, ctx.EvNumber, mtrack);
          m_outputTree->Fill();
        }
      }

 private:
      /// The config class
      MyMaterialTrackWriter_cfg m_cfg;
      /// The output file name
      TFile* m_outputFile = nullptr;
      /// The output tree name
      TTree* m_outputTree = nullptr;
      /// The read - write payload
      std::unique_ptr<ActsPlugins::RootMaterialTrackIo> m_payload;
};
// ##########################################################################################

std::vector<unsigned int> getExeptions(){

  std::vector<unsigned int> IDs_vec;
  std::ifstream file("/afs/cern.ch/user/l/ldamenti/CMSSW_16_0_0_pre1/src/ACTSinCMSSW/GeometryBuilder/python/Debug_test/oldRotModules.txt");
  std::string line;

  while (std::getline(file, line)) {
      const std::string marker = "Rotation matrices for ";
      std::size_t start = line.find(marker);
      if (start != std::string::npos) {
          start += marker.size();
          std::size_t end = line.find(" (TID)", start);
          if (end != std::string::npos) {
              unsigned int value = std::stoul(line.substr(start, end - start));
              IDs_vec.push_back(value);
          }
      }
  }

  return IDs_vec;
}

bool isTranspose(const Eigen::Matrix3d& A, const Eigen::Matrix3d& B, double eps = 1e-3) {
    return A.isApprox(B.transpose(), eps);
}

bool isEqual(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, double eps = 1e-3) {
    return A.isApprox(B, eps);
}

bool isOpposite(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, double eps = 1e-3) {
    if (A.rows() != B.rows() || A.cols() != B.cols())
        return false;

    return A.isApprox(-B, eps);
}

bool isNegativeTranspose(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, double eps = 1e-3) {
    if (A.rows() != B.cols() || A.cols() != B.rows())
        return false;

    return A.isApprox(-B.transpose(), eps);
}

bool isInverse(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, double eps = 1e-3) {
    if (A.rows() != A.cols() || B.rows() != B.cols() || A.rows() != B.rows())
        return false;

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(A.rows(), A.cols());
    return (A * B).isApprox(I, eps) && (B * A).isApprox(I, eps);
}


class ACTSTrackingGeometryProducer : public edm::ESProducer {
public:
  explicit ACTSTrackingGeometryProducer(const edm::ParameterSet& ps);
  ~ACTSTrackingGeometryProducer() override = default;
  //std::unique_ptr<Acts::TrackingGeometry> produce(const ACTSTrackerGeometryRecord& iRecord);  // shared to the vector of detector element
  std::shared_ptr<TrackingGeometryWithDetEls> produce(const ACTSTrackerGeometryRecord& iRecord);

private:
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopoToken_;
  edm::ESGetToken<Alignments, TrackerAlignmentRcd> trackerAlignToken_;

  bool saveObjfile_, saveSvgfile_, mapMaterial_;
  std::string outputObjFile_, outputSvgFile_, materialFile_;
  std::vector<double> rangeZ_;
  std::vector<double> rangeR_;
};


ACTSTrackingGeometryProducer::ACTSTrackingGeometryProducer(const edm::ParameterSet& ps)
    : saveObjfile_(ps.getUntrackedParameter<bool>("saveObjfile")),
      saveSvgfile_(ps.getUntrackedParameter<bool>("saveSvgfile")), 
      mapMaterial_(ps.getUntrackedParameter<bool>("mapMaterial")), 
      outputObjFile_(ps.getUntrackedParameter<std::string>("outputObjFile")),
      outputSvgFile_(ps.getUntrackedParameter<std::string>("outputSvgFile")),
      materialFile_(ps.getUntrackedParameter<std::string>("MaterialMaps")),
      rangeZ_(ps.getUntrackedParameter<std::vector<double>>("rangeZ")),
      rangeR_(ps.getUntrackedParameter<std::vector<double>>("rangeR")) {

    auto cc = setWhatProduced(this);
    trackerGeomToken_ = cc.consumes();
    trackerTopoToken_ = cc.consumes();
    trackerAlignToken_ = cc.consumes();
  }

//std::unique_ptr<Acts::TrackingGeometry> ACTSTrackingGeometryProducer::produce(const ACTSTrackerGeometryRecord& iRecord) { 
std::shared_ptr<TrackingGeometryWithDetEls> ACTSTrackingGeometryProducer::produce(const ACTSTrackerGeometryRecord& iRecord) { 

  const TrackerGeometry& trackerGeom = iRecord.get(trackerGeomToken_);
  const TrackerTopology& trackerTopo = iRecord.get(trackerTopoToken_);
  const Alignments& trackerAlign = iRecord.get(trackerAlignToken_);
  
  DetElVect DetEl_vector;

  // Get the modules with a different rotation scheme:
  std::vector<unsigned int> modulesNotTransposeRot = getExeptions();
   
  std::map<unsigned int, HepGeom::Transform3D> detID_to_alignInfo;
  // DEBUG
  // for (const auto& ali : trackerAlign.m_align) { 
  //   unsigned int rawID = static_cast<unsigned int>(ali.rawId());
  //   auto trs = ali.transform();
  //   detID_to_alignInfo[rawID] = trs; 
  //   std::cout << "[ALIGMENT] ALIGMENT INFO found for " << rawID << " -> translation: " << trs.getTranslation() << "; rotation: " << trs.getRotation() << std::endl;
  // }

  const Local2DPoint center(0.,0.); 
  const Local3DPoint locz(0.,0.,1.);
  const Local3DPoint locx(1.,0.,0.);
  const Local3DPoint locy(0.,1.,0.);
  const GlobalPoint origin(0.,0.,0.);
  
  // ===== Loop over all the CMSSW detector elments =====
  // for (const auto& det : trackerGeom.dets()) { // <-- DetElements might contain NOT physical surfaces (is an assemply of DetElementUnits) that's why I loop on DetElementUnits (sensitive surfaces ONLY)
  for (const auto& det : trackerGeom.detUnits()) {

    // ===== Get the surface described with CMSSW =====
    auto cmssw_surf = det->surface();

    // DEBUG pront the normal to the surface:
    // std::cout << "[CMSSW] Normal to module " << det->geographicalId().rawId() << ": " << cmssw_surf.normalVector() << std::endl;

    // ===== Define the transformation (i.e. Rotation and Translation) of the CMSSW surface =====
    auto& pos = cmssw_surf.position();
    auto& rot = cmssw_surf.rotation();
    Acts::Transform3 t = Acts::Transform3::Identity();
    Acts::RotationMatrix3 R;

    // ===== Get the Alignment INFO ===== (not needed as the aligment info are already stored in the transformation)
    DetId ID = det->geographicalId();
    // CLHEP::HepRotation Align_rot;
    // CLHEP::Hep3Vector Align_pos;
    // auto it = detID_to_alignInfo.find(ID.rawId());
    // if( it != detID_to_alignInfo.end()){
    //   auto this_tr = it->second;
    //   Align_rot = this_tr.getRotation();
    //   Align_pos = this_tr.getTranslation();
    //   DEBUG: compare rot e trs from surface and from alignm info
    //   std::cout << "-----------------\n"
    //             << "Module ID: " <<  ID.rawId() << "\n"
    //             << "Rotation and Translation from surface -> Rotation: " << rot << "; translation: " << pos << "\n"
    //             << "Rotation and Translation from alignment -> Rotation: " << Align_rot << "; translation: " << Align_pos << "\n" << std::endl;
    // }
    // else {
    //   std::cout << "[ERROR] No alignment info found for module " << ID.rawId() << std::endl;
    // }
    R << rot.xx(), rot.yx(), rot.zx(), rot.xy(), rot.yy(), rot.zy(), rot.xz(), rot.yz(), rot.zz(); // OLD
    // R << rot.xx(), rot.xy(), rot.xz(), rot.yx(), rot.yy(), rot.yz(), rot.zx(), rot.zy(), rot.zz(); // Transpose (Overlap in r)

    // ===== Try a different method to get R and t =====
    GlobalPoint position = trackerGeom.idToDet(ID)->toGlobal(center);
    GlobalPoint zpos = trackerGeom.idToDet(ID)->toGlobal(locz);
    GlobalPoint xpos = trackerGeom.idToDet(ID)->toGlobal(locx);
    GlobalPoint ypos = trackerGeom.idToDet(ID)->toGlobal(locy);
    GlobalVector dz = zpos - position;
    GlobalVector dx = xpos - position;
    GlobalVector dy = ypos - position;

    Eigen::Vector3d dxV(dx.x(), dx.y(), dx.z());
    Eigen::Vector3d dyV(dy.x(), dy.y(), dy.z());
    Eigen::Vector3d dVz(dz.x(), dz.y(), dz.z());

    dxV.normalize();
    dyV.normalize();
    dxV = dxV - dxV.dot(dVz) * dVz; // make dxV ortogonal to dz
    dVz.normalize();

    dyV = dVz.cross(dxV);  // make dy ortogonal to dx e dz

    Eigen::Matrix3d Rot;
    Rot.col(0) = dxV;
    Rot.col(1) = dyV;
    Rot.col(2) = dVz;

    std::string subDet;
    if(ID.subdetId() == PixelSubdetector::PixelBarrel) {
      subDet = "PixelBarrel";
    } 
    else if (ID.subdetId() == PixelSubdetector::PixelEndcap) {
      subDet = "PixelEndcap";
    }     
    else if (ID.subdetId() == StripSubdetector::TID) {
      subDet = "TID";
    }     
    else if (ID.subdetId() == StripSubdetector::TOB) {
      subDet = "TOB";
    }     
    else if (ID.subdetId() == StripSubdetector::TIB) {
      subDet = "TIB";
    }     
    else if (ID.subdetId() == StripSubdetector::TEC)  {
      subDet = "TEC";
    }    

    t.prerotate(Rot);
    t.pretranslate(Acts::Vector3(position.x()*10, position.y()*10, position.z()*10)); // from cm to mm

    // if(isTranspose(R, Rot)){
    //   std::cout << subDet << ", " <<  det->geographicalId().rawId() << ", B is the transpose of A" << std::endl; 
    // }
    // else if (isEqual(R, Rot)){
    //   std::cout << subDet << ", " <<  det->geographicalId().rawId() << ", B is Equal to of A" << std::endl; 

    // }
    // else if (isOpposite(R, Rot)){
    //   std::cout << subDet << ", " <<  det->geographicalId().rawId() << ", B is Opposite to of A" << std::endl; 

    // }
    // else if (isNegativeTranspose(R, Rot)){
    //   std::cout << subDet << ", " <<  det->geographicalId().rawId() << ", B is the negative transpose to of A" << std::endl; 

    // }
    // else if (isInverse(R, Rot)){
    //   std::cout << subDet << ", " <<  det->geographicalId().rawId() << ", B is the inverse of A" << std::endl; 

    // }
    // else {
    //   std::cout << subDet << ", " <<  det->geographicalId().rawId() << ", none relations found for A and B" << std::endl; 
    // }


    // ===== Is this detector element unit flipped? =====
    // const GlobalPoint Pos = cmssw_surf.position();
    // const GlobalVector nrm = cmssw_surf.normalVector();
    // const bool isLower = trackerTopo.isLower(ID);
    // // const bool normalOut = innerOuterFromOrientation(trackerTopo, ID, Pos, nrm);
    // const int innerOuter = innerOuterFromOrientation(trackerTopo, ID, Pos, nrm);
    // //const bool isFlipped = ( isLower && innerOuter == 1 ) || ( !isLower && innerOuter == 0 );    

    // Eigen::Matrix3d Rz;
    // Rz << -1,  0,  0,
    //        0, -1,  0,
    //        0,  0,  1;
    // Eigen::Matrix3d FlipX;
    // FlipX << -1,  0,  0,
    //            0,  1,  0,
    //            0,  0,  1;
    // Eigen::Matrix3d FlipY;
    // FlipY <<   1,  0,  0,
    //            0, -1,  0,
    //            0,  0,  1;

    // if (ID.subdetId() == StripSubdetector::TIB) {
    //   std::cout << "TIB module found: layer = " << trackerTopo.layer(ID) << "; side = " << trackerTopo.tibSide(ID) << "; Order = " << trackerTopo.tibOrder(ID) << std::endl;


    //   R << rot.xx(), rot.yx(), rot.zx(), rot.xy(), rot.yy(), rot.zy(), rot.xz(), rot.yz(), rot.zz(); // OLD

    //   if(trackerTopo.tibIsStereo(ID)){
    //     double alpha = 0.100;
    //     alpha *= trackerTopo.tibSide(ID) * ((trackerTopo.tibString(ID) % 2 == 0) ? +1 : -1); 
    //     std::cout << "TIB module is stereo (alpha = " << alpha << ")" << std::endl;
    //     Eigen::Matrix3d Rstereo;
    //     Rstereo << cos(alpha), -sin(alpha), 0,
    //                sin(alpha),  cos(alpha), 0,
    //               0,           0,          1;

    //     R = R * Rstereo;
    //   }


    //   if (innerOuter == 0) {
    //     std::cout << "Module is flipped, rotating it" << std::endl;
    //     // R << rot.xx(), -rot.yx(), -rot.zx(), rot.xy(), -rot.yy(), -rot.zy(), rot.xz(), -rot.yz(), -rot.zz(); // <-- Flip Z and Y
    //     // R << -rot.xx(), -rot.yx(), -rot.zx(), -rot.xy(), -rot.yy(), -rot.zy(), -rot.xz(), -rot.yz(), -rot.zz(); // <-- Flip Z, Y and X 
    //     // R << -rot.xx(), -rot.yx(), -rot.zx(), -rot.xy(), -rot.yy(), -rot.zy(),  rot.xz(),  rot.yz(),  rot.zz(); // <-- 180 deg rot around Z 
    //     R << rot.xx(), rot.yx(), rot.zx(), rot.xy(), rot.yy(), rot.zy(), rot.xz(), rot.yz(), rot.zz(); // OLD
    //     // R <<  rot.xx(),  rot.yx(), -rot.zx(), rot.xy(),  rot.yy(), -rot.zy(), -rot.xz(), -rot.yz(),  rot.zz();
    //     // R <<  rot.xx(), -rot.yx(),  rot.zx(), rot.xy(), -rot.yy(),  rot.zy(), -rot.xz(),  rot.yz(), -rot.zz();
    //     R = Rz * R * FlipX;
    //   }
    //   else {
    //     R << rot.xx(), rot.yx(), rot.zx(), rot.xy(), rot.yy(), rot.zy(), rot.xz(), rot.yz(), rot.zz(); // OLD
    //   }
    // }
    // else if(ID.subdetId() == StripSubdetector::TOB) {
    //   if (innerOuter == 0) {
    //     R << rot.xx(), rot.yx(), rot.zx(), rot.xy(), rot.yy(), rot.zy(), rot.xz(), rot.yz(), rot.zz(); // OLD
    //     R = R * FlipY;
    //   }
    //   else {
    //     R << rot.xx(), rot.yx(), rot.zx(), rot.xy(), rot.yy(), rot.zy(), rot.xz(), rot.yz(), rot.zz(); // OLD
    //   }
    // }
    // else if (ID.subdetId() == StripSubdetector::TEC) {
    //   if (innerOuter == 0) {
    //     R << rot.xx(), rot.yx(), rot.zx(), rot.xy(), rot.yy(), rot.zy(), rot.xz(), rot.yz(), rot.zz(); // OLD
    //   }
    //   else {
    //     R << rot.xx(), rot.yx(), rot.zx(), rot.xy(), rot.yy(), rot.zy(), rot.xz(), rot.yz(), rot.zz(); // OLD
    //   }
    // }
    // else {
    //   R << rot.xx(), rot.yx(), rot.zx(), rot.xy(), rot.yy(), rot.zy(), rot.xz(), rot.yz(), rot.zz(); // OLD
    // }
    // else if (ID.subdetId() == StripSubdetector::TOB){

    //   if ((trackerTopo.tobLayer(ID) == 1 && trackerTopo.tobIsStereo(ID)) || (trackerTopo.tobLayer(ID) == 2)){// && trackerTopo.tobIsStereo(ID))){
    //     R << rot.xx(), -rot.yx(), -rot.zx(), rot.xy(), -rot.yy(), -rot.zy(), rot.xz(), -rot.yz(), -rot.zz();
    //   }      
    //   else {
    //     R << rot.xx(), rot.yx(), rot.zx(), rot.xy(), rot.yy(), rot.zy(), rot.xz(), rot.yz(), rot.zz(); // OLD
    //   }
    // }


    // R << rot.xx(), rot.xy(), rot.xz(), rot.yx(), rot.yy(), rot.yz(), rot.zx(), rot.zy(), rot.zz(); // right-multiplication (no)
    // R << rot.xx(), rot.xy(), rot.xz(), rot.yx(), rot.yy(), rot.yz(), rot.zx(), rot.zy(), rot.zz(); // Trasposta (no)
    // R << rot.xx(), rot.xy(), -rot.xz(), rot.yx(), rot.yy(), -rot.yz(), rot.zx(), rot.zy(), -rot.zz(); // Z filePath (no)
    // R << rot.xy(), rot.yy(), rot.zy(), rot.xx(), rot.yx(), rot.zx(), rot.xz(), rot.yz(), rot.zz(); // swap X Y (no)

    // auto detID = det->geographicalId().rawId();
    // if (std::find(modulesNotTransposeRot.begin(), modulesNotTransposeRot.end(), detID) != modulesNotTransposeRot.end()){
    //   R << rot.xx(), rot.xy(), rot.xz(), rot.yx(), rot.yy(), rot.yz(), rot.zx(), rot.zy(), rot.zz(); // equal to the one from cmssw (for a subset of modules from TEC and TID)
    // }
    // else {
    //   R << rot.xx(), rot.yx(), rot.zx(), rot.xy(), rot.yy(), rot.zy(), rot.xz(), rot.yz(), rot.zz(); // OLD (transposed wrt cmssw one)
    // }

    // ===== Define the ACTS surface considering two tipes of bounds (i.e. Rectangle and Trapezoid) =====
    std::shared_ptr<Acts::Surface> acts_surf = nullptr;
    auto bounds = dynamic_cast<const RectangularPlaneBounds*>(&cmssw_surf.bounds());
    if (bounds){
      const std::size_t kValues = Acts::RectangleBounds::BoundValues::eSize;
      std::array<double, kValues> bValues{};
      std::vector<double> bVector = {-bounds->width()  / 2 * 10,  // cm → mm
                                     -bounds->length() / 2 * 10,
                                      bounds->width()  / 2 * 10,
                                      bounds->length() / 2 * 10};

      std::copy_n(bVector.begin(), kValues, bValues.begin());
      acts_surf = Acts::Surface::makeShared<Acts::PlaneSurface>(t, std::move(std::make_shared<const Acts::RectangleBounds>(bValues)));
    }
    else{
      auto trap = dynamic_cast<const TrapezoidalPlaneBounds*>(&cmssw_surf.bounds());
      auto params = trap->parameters();

      const std::size_t kValues = Acts::TrapezoidBounds::BoundValues::eSize;
      std::array<double, kValues> bValues{};
      // OLD WRONG WAY
      // std::vector<double> bVector = {params[3] * 10, // cm to mm
      //                                params[3] * 10, 
      //                                (params[0] + params[1]) / 2 * 10, 
      //                                0.0};
      std::vector<double> bVector = { params[0] * 10,  // half bottom edge 
                                      params[1] * 10,  // half top edge    
                                      params[3] * 10,  // half Y / apothem 
                                      0.0};               // rotation angle

      std::copy_n(bVector.begin(), kValues, bValues.begin());
      acts_surf = Acts::Surface::makeShared<Acts::PlaneSurface>(t, std::move(std::make_shared<const Acts::TrapezoidBounds>(bValues)));

    }

    // ===== Define the CMSDetectorElementData (i.e. the struct needed to define the CMS Detector Element in ACTS) =====
    Acts::CMSDetectorElementData cmsDetData;
    cmsDetData.surf_ = acts_surf;
    cmsDetData.trans_ = t;

    DetId detid = det->geographicalId();
    cmsDetData.detID_ = detid.rawId();
    
    if (detid.subdetId() == PixelSubdetector::PixelBarrel) {
      cmsDetData.subDetector_ = std::string("PixelBarrel");
      cmsDetData.thickness_ = 0.285*Acts::UnitConstants::mm;
    } else if (detid.subdetId() == PixelSubdetector::PixelEndcap) {
      cmsDetData.subDetector_ = std::string("PixelEndcap");
      cmsDetData.thickness_ = 0.3*Acts::UnitConstants::mm;
    } else if (detid.subdetId() == StripSubdetector::TIB) {
      cmsDetData.subDetector_ = std::string("TIB");
      cmsDetData.thickness_ = 0.32*Acts::UnitConstants::mm;
    } else if (detid.subdetId() == StripSubdetector::TID) {
      cmsDetData.subDetector_ = std::string("TID");
      cmsDetData.thickness_ = 0.32*Acts::UnitConstants::mm;
    } else if (detid.subdetId() == StripSubdetector::TOB) {
      cmsDetData.subDetector_ = std::string("TOB");
      cmsDetData.thickness_ = 0.5*Acts::UnitConstants::mm;
    } else if (detid.subdetId() == StripSubdetector::TEC) {
      cmsDetData.subDetector_ = std::string("TEC");
      cmsDetData.thickness_ = 0.5*Acts::UnitConstants::mm;
    }

    // ===== Save the converted detector element in a vector =====
    DetEl_vector.push_back(std::make_shared<Acts::CMSDetectorElement>(cmsDetData));

  }

  // Collect all the surfaces of the detector elements:
  std::vector<std::shared_ptr<Acts::Surface>> surface_vector;
  for(auto DetEl : DetEl_vector){
    surface_vector.push_back((*DetEl).surface().getSharedPtr());
  }

  // Save all the surfaces in a Kdt vector:
  const Acts::GeometryContext viewContext;
  KdtSurfacesDim2Bin100 Kdtsurfaces(viewContext, surface_vector, casts);


  // ===== OPTIONAL: Save the surfaces associate to the detector elements in an obj file =====
  if(saveObjfile_) {

    std::cout << ">>> Flag saveObjfile True <<<" << std::endl;
    std::cout << "Saving detector elements as obj file in the range: " << std::endl;
    std::cout << "R = [" << rangeR_[0] << ", " << rangeR_[1] << "]" << std::endl;
    std::cout << "Z = [" << rangeZ_[0] << ", " << rangeZ_[1] << "]" << std::endl;

    // Select a subset of surfaces:
    std::array<double, 2> min = {rangeZ_[0],rangeR_[0]};
    std::array<double, 2> max = {rangeZ_[1],rangeR_[1]};
    Acts::RangeXD<2, double, std::array> RangeClass(min, max);
    std::vector<std::shared_ptr<Acts::Surface>> surfaceQuery = Kdtsurfaces.surfaces(RangeClass);

    writeSurfacesObj(surfaceQuery, outputObjFile_); 

  }

  // ===== Save all the sensitive surfaces in a JSON file =====
  Acts::GeometryContext geoC;
  std::vector<nlohmann::json> jSurfaces;
  for (const auto& sSurface : surface_vector) {
      jSurfaces.push_back(Acts::SurfaceJsonConverter::toJson(geoC, *sSurface));
  }

  nlohmann::json jSurfacesAll;
  jSurfacesAll["surfaces"] = jSurfaces;

  std::cout << ">>> Storing all the surfaces into a json file <<<" << std::endl;
  std::ofstream file("CMSPhaseI_Sensitive_All.json");
  file << jSurfacesAll.dump(4) << '\n';


  // ===== Make the blueprint =====
  Acts::Transform3 base{Acts::Transform3::Identity()};

  Acts::Experimental::Blueprint::Config cfg;
  cfg.envelope[Acts::AxisDirection::AxisZ] = {5*Acts::UnitConstants::mm, 5*Acts::UnitConstants::mm};
  cfg.envelope[Acts::AxisDirection::AxisR] = {10*Acts::UnitConstants::mm, 20*Acts::UnitConstants::mm};
  auto root = std::make_unique<Acts::Experimental::Blueprint>(cfg);

  root->addMaterial("GlobalMaterial", [&](Acts::Experimental::MaterialDesignatorBlueprintNode& mat) {
    using enum Acts::AxisDirection;
    using enum Acts::AxisBoundaryType;
    using enum Acts::CylinderVolumeBounds::Face;

    // Configure cylinder faces with proper binning
    mat.configureFace(Acts::CylinderVolumeBounds::Face::OuterCylinder, {Acts::AxisDirection::AxisRPhi, Acts::AxisBoundaryType::Bound, 20}, {Acts::AxisDirection::AxisZ, Acts::AxisBoundaryType::Bound, 20});
    mat.configureFace(Acts::CylinderVolumeBounds::Face::NegativeDisc, {Acts::AxisDirection::AxisR, Acts::AxisBoundaryType::Bound, 15}, {Acts::AxisDirection::AxisPhi, Acts::AxisBoundaryType::Bound, 25});
    mat.configureFace(Acts::CylinderVolumeBounds::Face::PositiveDisc, {Acts::AxisDirection::AxisR, Acts::AxisBoundaryType::Bound, 15}, {Acts::AxisDirection::AxisPhi, Acts::AxisBoundaryType::Bound, 25});

    mat.addCylinderContainer("CMS", Acts::AxisDirection::AxisR, [&](auto& CMS) {
      CMS.setAttachmentStrategy(Acts::VolumeAttachmentStrategy::Gap);

      CMS.addCylinderContainer("SubDetectors", Acts::AxisDirection::AxisZ, [&](auto& det) {
        det.setAttachmentStrategy(Acts::VolumeAttachmentStrategy::Gap);

        det.addCylinderContainer("NegativeTEC", Acts::AxisDirection::AxisZ, [&](auto& ec) {
              ec.setAttachmentStrategy(Acts::VolumeAttachmentStrategy::Gap);
              //.setResizeStrategy(Acts::VolumeResizeStrategy::Gap);

              auto makeLayer = [&](const Acts::Transform3& trf, Acts::Experimental::LayerBlueprintNode& layer, std::string disk_name, double Bin_Phi, double Bin_R) {
                std::vector<std::shared_ptr<Acts::Surface>> surfaces_disk = SelectActiveSurfaces_PhaseI(Kdtsurfaces, disk_name);
                // Binning:
                makeBinning(layer, Acts::SurfaceArrayNavigationPolicy::LayerType::Disc, Bin_Phi, Bin_R);

                layer.setSurfaces(surfaces_disk)
                    .setLayerType(Acts::Experimental::LayerBlueprintNode::LayerType::Disc)
                    .setEnvelope(Acts::ExtentEnvelope{{
                        .z = {5*Acts::UnitConstants::mm, 5*Acts::UnitConstants::mm},
                        .r = {10*Acts::UnitConstants::mm, 20*Acts::UnitConstants::mm},
                    }})
                    .setTransform(base)
                    .setUseCenterOfGravity(false, false, true); // To fix the transaltion on x-y
              };

              AddDiskLayer_and_Material(&ec, "NegTEC0", makeLayer, -1320*Acts::UnitConstants::mm, 80, 4);
              AddDiskLayer_and_Material(&ec, "NegTEC1", makeLayer, -1463*Acts::UnitConstants::mm, 80, 5);
              AddDiskLayer_and_Material(&ec, "NegTEC2", makeLayer, -1603*Acts::UnitConstants::mm, 80, 5);
              AddDiskLayer_and_Material(&ec, "NegTEC3", makeLayer, -1744*Acts::UnitConstants::mm, 80, 6);
              AddDiskLayer_and_Material(&ec, "NegTEC4", makeLayer, -1883*Acts::UnitConstants::mm, 80, 6);
              AddDiskLayer_and_Material(&ec, "NegTEC5", makeLayer, -2058*Acts::UnitConstants::mm, 80, 6);
              AddExtraLayer("ExtraECMatLater4", false, &ec, GenerateTranslation(0, 0, -2150*Acts::UnitConstants::mm), std::make_shared<Acts::CylinderVolumeBounds>(10*Acts::UnitConstants::mm, 1200*Acts::UnitConstants::mm, 10*Acts::UnitConstants::mm));
              AddDiskLayer_and_Material(&ec, "NegTEC6", makeLayer, -2248*Acts::UnitConstants::mm, 80, 7);
              AddExtraLayer("ExtraECMatLater5", false, &ec, GenerateTranslation(0, 0, -2350*Acts::UnitConstants::mm), std::make_shared<Acts::CylinderVolumeBounds>(10*Acts::UnitConstants::mm, 1200*Acts::UnitConstants::mm, 10*Acts::UnitConstants::mm));
              AddDiskLayer_and_Material(&ec, "NegTEC7", makeLayer, -2454*Acts::UnitConstants::mm, 80, 7);
              AddExtraLayer("ExtraECMatLater6", false, &ec, GenerateTranslation(0, 0, -2550*Acts::UnitConstants::mm), std::make_shared<Acts::CylinderVolumeBounds>(10*Acts::UnitConstants::mm, 1200*Acts::UnitConstants::mm, 10*Acts::UnitConstants::mm));

              ec.addLayer("NegTEC8", [&](auto& layer) {
                makeLayer(base * Acts::Translation3{Acts::Vector3{0, 0, -2666*Acts::UnitConstants::mm}}, layer, "NegTEC8", 80, 7);
              });


            });
        
        AddExtraLayer("extra_ecneg", false, &det, GenerateTranslation(0, 0, -1150*Acts::UnitConstants::mm), std::make_shared<Acts::CylinderVolumeBounds>(10*Acts::UnitConstants::mm, 1200*Acts::UnitConstants::mm, 10*Acts::UnitConstants::mm));
        AddExtraLayer("extra_ecneg2", false, &det, GenerateTranslation(0, 0, -1200*Acts::UnitConstants::mm), std::make_shared<Acts::CylinderVolumeBounds>(10*Acts::UnitConstants::mm, 1200*Acts::UnitConstants::mm, 10*Acts::UnitConstants::mm));
                    
        det.addCylinderContainer("Pixel_TIB_TID_TOB", Acts::AxisDirection::AxisR, [&](auto& barr) { 
          barr.setAttachmentStrategy(Acts::VolumeAttachmentStrategy::Gap)
                .setResizeStrategy(Acts::VolumeResizeStrategy::Gap);

          barr.addCylinderContainer("Pixel", Acts::AxisDirection::AxisZ, [&](auto& cyl) { 
            cyl.setAttachmentStrategy(Acts::VolumeAttachmentStrategy::Gap)
                .setResizeStrategy(Acts::VolumeResizeStrategy::Gap);
            
            AddExtraLayer("extra_after_frwPixelNeg1", false, &cyl, GenerateTranslation(0, 0, -800*Acts::UnitConstants::mm), std::make_shared<Acts::CylinderVolumeBounds>(10*Acts::UnitConstants::mm, 200*Acts::UnitConstants::mm, 10*Acts::UnitConstants::mm));
            AddExtraLayer("extra_after_frwPixelNeg2", false, &cyl, GenerateTranslation(0, 0, -700*Acts::UnitConstants::mm), std::make_shared<Acts::CylinderVolumeBounds>(10*Acts::UnitConstants::mm, 200*Acts::UnitConstants::mm, 10*Acts::UnitConstants::mm));
            AddExtraLayer("extra_after_frwPixelNeg3", false, &cyl, GenerateTranslation(0, 0, -600*Acts::UnitConstants::mm), std::make_shared<Acts::CylinderVolumeBounds>(10*Acts::UnitConstants::mm, 200*Acts::UnitConstants::mm, 10*Acts::UnitConstants::mm));
             
            cyl.addCylinderContainer(
                "PixelNegativeEndcap", Acts::AxisDirection::AxisZ, [&](auto& ec) {
                  ec.setAttachmentStrategy(Acts::VolumeAttachmentStrategy::Gap);

                  auto makeLayer = [&](const Acts::Transform3& trf, auto& layer, std::string disk_name, double Bin_Phi, double Bin_R) {
                    std::vector<std::shared_ptr<Acts::Surface>> surfaces_disk = SelectActiveSurfaces_PhaseI(Kdtsurfaces, disk_name);
                    // Binning:
                    makeBinning(layer, Acts::SurfaceArrayNavigationPolicy::LayerType::Disc, Bin_Phi, Bin_R);

                    layer.setSurfaces(surfaces_disk)
                        .setLayerType(Acts::Experimental::LayerBlueprintNode::LayerType::Disc)
                        .setEnvelope(Acts::ExtentEnvelope{{
                            .z = {5*Acts::UnitConstants::mm, 5*Acts::UnitConstants::mm},
                            .r = {10*Acts::UnitConstants::mm, 20*Acts::UnitConstants::mm},
                        }})
                        .setTransform(base)
                        .setUseCenterOfGravity(false, false, true); // To fix the transaltion on x-y
                  };

                  AddDiskLayer_and_Material(&ec, "PixelNeg0", makeLayer, -322*Acts::UnitConstants::mm, 36, 2);
                  AddDiskLayer_and_Material(&ec, "PixelNeg1", makeLayer, -395*Acts::UnitConstants::mm, 36, 2);
                  AddDiskLayer_and_Material(&ec, "PixelNeg2", makeLayer, -493*Acts::UnitConstants::mm, 36, 2);

                });
            
            cyl.addCylinderContainer(
                "PixelBarrel", Acts::AxisDirection::AxisR, [&](auto& brl) {
                  brl.setAttachmentStrategy(Acts::VolumeAttachmentStrategy::Gap)
                      .setResizeStrategy(Acts::VolumeResizeStrategy::Gap);


                  AddCylinderLayer_and_Material(&brl, "PixL0", Kdtsurfaces, 12, 8);
                  AddCylinderLayer_and_Material(&brl, "PixL1", Kdtsurfaces, 32, 8);
                  AddCylinderLayer_and_Material(&brl, "PixL2", Kdtsurfaces, 48, 8);
                  AddCylinderLayer_and_Material(&brl, "PixL3", Kdtsurfaces, 64, 8);
                });
            
            cyl.addCylinderContainer(
                "PixelPositiveEndcap", Acts::AxisDirection::AxisZ, [&](auto& ec) {
                  ec.setAttachmentStrategy(Acts::VolumeAttachmentStrategy::Gap);

                  auto makeLayer = [&](const Acts::Transform3& trf, auto& layer, std::string disk_name, double Bin_Phi, double Bin_R) {
                    std::vector<std::shared_ptr<Acts::Surface>> surfaces_disk = SelectActiveSurfaces_PhaseI(Kdtsurfaces, disk_name);
                    // Binning:
                    makeBinning(layer, Acts::SurfaceArrayNavigationPolicy::LayerType::Disc, Bin_Phi, Bin_R);

                    layer.setSurfaces(surfaces_disk)
                        .setLayerType(Acts::Experimental::LayerBlueprintNode::LayerType::Disc)
                        .setEnvelope(Acts::ExtentEnvelope{{
                            .z = {5*Acts::UnitConstants::mm, 5*Acts::UnitConstants::mm},
                            .r = {10*Acts::UnitConstants::mm, 20*Acts::UnitConstants::mm},
                        }})
                        .setTransform(base)
                        .setUseCenterOfGravity(false, false, true); // To fix the transaltion on x-y
                  };

                  AddDiskLayer_and_Material(&ec, "PixelPos0", makeLayer, 322*Acts::UnitConstants::mm, 36, 2);
                  AddDiskLayer_and_Material(&ec, "PixelPos1", makeLayer, 395*Acts::UnitConstants::mm, 36, 2);
                  AddDiskLayer_and_Material(&ec, "PixelPos2", makeLayer, 493*Acts::UnitConstants::mm, 36, 2);
                });

            AddExtraLayer("extra_after_frwPixelPos1", false, &cyl, GenerateTranslation(0, 0, 600*Acts::UnitConstants::mm), std::make_shared<Acts::CylinderVolumeBounds>(10*Acts::UnitConstants::mm, 200*Acts::UnitConstants::mm, 10*Acts::UnitConstants::mm));
            AddExtraLayer("extra_after_frwPixelPos2", false, &cyl, GenerateTranslation(0, 0, 700*Acts::UnitConstants::mm), std::make_shared<Acts::CylinderVolumeBounds>(10*Acts::UnitConstants::mm, 200*Acts::UnitConstants::mm, 10*Acts::UnitConstants::mm));
            AddExtraLayer("extra_after_frwPixelPos3", false, &cyl, GenerateTranslation(0, 0, 800*Acts::UnitConstants::mm), std::make_shared<Acts::CylinderVolumeBounds>(10*Acts::UnitConstants::mm, 200*Acts::UnitConstants::mm, 10*Acts::UnitConstants::mm));
              
            });

          barr.addCylinderContainer("TIB_TID", Acts::AxisDirection::AxisZ, [&](auto& cyl) {
            cyl.setAttachmentStrategy(Acts::VolumeAttachmentStrategy::Gap)
                .setResizeStrategy(Acts::VolumeResizeStrategy::Gap);
            
            cyl.addCylinderContainer(
                "NegativeTID", Acts::AxisDirection::AxisZ, [&](auto& ec) {
                  ec.setAttachmentStrategy(Acts::VolumeAttachmentStrategy::Gap);

                  auto makeLayer = [&](const Acts::Transform3& trf, auto& layer, std::string disk_name, double Bin_Phi, double Bin_R) {
                    std::vector<std::shared_ptr<Acts::Surface>> surfaces_disk = SelectActiveSurfaces_PhaseI(Kdtsurfaces, disk_name);
                    // Binning:
                    makeBinning(layer, Acts::SurfaceArrayNavigationPolicy::LayerType::Disc, Bin_Phi, Bin_R);

                    layer.setSurfaces(surfaces_disk)
                        .setLayerType(Acts::Experimental::LayerBlueprintNode::LayerType::Disc)
                        .setEnvelope(Acts::ExtentEnvelope{{
                            .z = {5*Acts::UnitConstants::mm, 5*Acts::UnitConstants::mm},
                            .r = {10*Acts::UnitConstants::mm, 20*Acts::UnitConstants::mm},
                        }})
                        .setTransform(base)
                        .setUseCenterOfGravity(false, false, true); // To fix the transaltion on x-y
                  };

                  AddDiskLayer_and_Material(&ec, "TIDNeg0", makeLayer, -790*Acts::UnitConstants::mm, 40, 3);
                  AddDiskLayer_and_Material(&ec, "TIDNeg1", makeLayer, -920*Acts::UnitConstants::mm, 40, 3);

                  ec.addLayer("TIDNeg2", [&](auto& layer) {
                    makeLayer(base * Acts::Translation3{Acts::Vector3{0, 0, -1050*Acts::UnitConstants::mm}}, layer, "TIDNeg2", 40, 3);
                  });

                });
              
            cyl.addCylinderContainer(
                "TIB", Acts::AxisDirection::AxisR, [&](auto& brl) {
                  brl.setAttachmentStrategy(Acts::VolumeAttachmentStrategy::Gap)
                      .setResizeStrategy(Acts::VolumeResizeStrategy::Gap);

                  auto makeLayer = [&](const std::string& name, double Bin_Phi, double Bin_Z) {
                    brl.addLayer(name, [&](auto& layer) {
                      std::vector<std::shared_ptr<Acts::Surface>> surfaces = SelectActiveSurfaces_PhaseI(Kdtsurfaces, name);
                      // Binning:
                      makeBinning(layer, Acts::SurfaceArrayNavigationPolicy::LayerType::Cylinder, Bin_Phi, Bin_Z);

                      layer.setSurfaces(surfaces)
                          .setLayerType(Acts::Experimental::LayerBlueprintNode::LayerType::Cylinder)
                          .setEnvelope(Acts::ExtentEnvelope{{
                              .z = {5*Acts::UnitConstants::mm, 5*Acts::UnitConstants::mm},
                              .r = {1*Acts::UnitConstants::mm, 1*Acts::UnitConstants::mm},
                          }})
                          .setTransform(base)
                          .setUseCenterOfGravity(false, false, false); // To fix the transaltion on x-y
                    });
                  };
                  
                  AddCylinderLayer_and_Material(&brl, "TIB0", Kdtsurfaces, 26, 10);
                  AddCylinderLayer_and_Material(&brl, "TIB1", Kdtsurfaces, 38, 10);
                  AddCylinderLayer_and_Material(&brl, "TIB2", Kdtsurfaces, 44, 10);

                  makeLayer("TIB3", 56,10);
                });
            
            cyl.addCylinderContainer(
                "PositiveTID", Acts::AxisDirection::AxisZ, [&](auto& ec) {
                  ec.setAttachmentStrategy(Acts::VolumeAttachmentStrategy::Gap);

                  auto makeLayer = [&](const Acts::Transform3& trf, auto& layer, std::string disk_name, double Bin_Phi, double Bin_R) {
                    std::vector<std::shared_ptr<Acts::Surface>> surfaces_disk = SelectActiveSurfaces_PhaseI(Kdtsurfaces, disk_name);
                    // Binning:
                    makeBinning(layer, Acts::SurfaceArrayNavigationPolicy::LayerType::Disc, Bin_Phi, Bin_R);

                    layer.setSurfaces(surfaces_disk)
                        .setLayerType(Acts::Experimental::LayerBlueprintNode::LayerType::Disc)
                        .setEnvelope(Acts::ExtentEnvelope{{
                            .z = {5*Acts::UnitConstants::mm, 5*Acts::UnitConstants::mm},
                            .r = {10*Acts::UnitConstants::mm, 20*Acts::UnitConstants::mm},
                        }})
                        .setTransform(base)
                        .setUseCenterOfGravity(false, false, true); // To fix the transaltion on x-y
                  };
                  //makeLayer(translation, layer_to_move, "Layer_Name", Bin_Phi, Bin_R)
                  AddDiskLayer_and_Material(&ec, "TIDPos0", makeLayer, 790*Acts::UnitConstants::mm, 40, 3);
                  AddDiskLayer_and_Material(&ec, "TIDPos1", makeLayer, 920*Acts::UnitConstants::mm, 40, 3);

                  ec.addLayer("TIDPos2", [&](auto& layer) {
                    makeLayer(base * Acts::Translation3{Acts::Vector3{0, 0, 1050*Acts::UnitConstants::mm}}, layer, "TIDPos2", 40, 3);
                  });
            });
            
          });

          AddExtraLayer("ExtraTIB_TOBMatLayer1", true, &barr, GenerateTranslation(0, 0, 0), std::make_shared<Acts::CylinderVolumeBounds>(530*Acts::UnitConstants::mm, 540*Acts::UnitConstants::mm, 1000*Acts::UnitConstants::mm));
          AddExtraLayer("ExtraTIB_TOBMatLayer2", true, &barr, GenerateTranslation(0, 0, 0), std::make_shared<Acts::CylinderVolumeBounds>(560*Acts::UnitConstants::mm, 570*Acts::UnitConstants::mm, 1000*Acts::UnitConstants::mm));

          barr.addCylinderContainer(
              "TOB", Acts::AxisDirection::AxisR, [&](auto& brl) {
                brl.setAttachmentStrategy(Acts::VolumeAttachmentStrategy::Gap)
                    .setResizeStrategy(Acts::VolumeResizeStrategy::Gap);

                // auto makeLayer = [&](const std::string& name, double Bin_Phi, double Bin_Z) {
                //     brl.addLayer(name, [&](auto& layer) {
                //       std::vector<std::shared_ptr<Acts::Surface>> surfaces = SelectActiveSurfaces_PhaseI(Kdtsurfaces, name);
                //       // Binning:
                //       makeBinning(layer, Acts::SurfaceArrayNavigationPolicy::LayerType::Cylinder, Bin_Phi, Bin_Z);

                //       layer.setSurfaces(surfaces)
                //           .setLayerType(Acts::Experimental::LayerBlueprintNode::LayerType::Cylinder)
                //           .setEnvelope(Acts::ExtentEnvelope{{
                //               .z = {5*Acts::UnitConstants::mm, 5*Acts::UnitConstants::mm},
                //               .r = {1*Acts::UnitConstants::mm, 1*Acts::UnitConstants::mm},
                //           }})
                //           .setTransform(base)
                //           .setUseCenterOfGravity(false, false, false); // To fix the transaltion on x-y
                //     });
                //   };

                // makeLayer("Layer_name", Bin_Phi, Bin_Z)
                AddCylinderLayer_and_Material(&brl, "TOB0", Kdtsurfaces, 42, 24);
                AddExtraLayer("ExtraTOBMatLayer1", true, &brl, GenerateTranslation(0, 0, 0), std::make_shared<Acts::CylinderVolumeBounds>(650*Acts::UnitConstants::mm, 655*Acts::UnitConstants::mm, 1000*Acts::UnitConstants::mm));
                AddCylinderLayer_and_Material(&brl, "TOB1", Kdtsurfaces, 48, 24);
                AddExtraLayer("ExtraTOBMatLayer2", true, &brl, GenerateTranslation(0, 0, 0), std::make_shared<Acts::CylinderVolumeBounds>(740*Acts::UnitConstants::mm, 750*Acts::UnitConstants::mm, 1000*Acts::UnitConstants::mm));
                AddCylinderLayer_and_Material(&brl, "TOB2", Kdtsurfaces, 54, 24);
                AddExtraLayer("ExtraTOBMatLayer3", true, &brl, GenerateTranslation(0, 0, 0), std::make_shared<Acts::CylinderVolumeBounds>(805*Acts::UnitConstants::mm, 810*Acts::UnitConstants::mm, 1000*Acts::UnitConstants::mm));
                AddCylinderLayer_and_Material(&brl, "TOB3", Kdtsurfaces, 60, 24);
                AddExtraLayer("ExtraTOBMatLayer4", true, &brl, GenerateTranslation(0, 0, 0), std::make_shared<Acts::CylinderVolumeBounds>(940*Acts::UnitConstants::mm, 943*Acts::UnitConstants::mm, 1000*Acts::UnitConstants::mm));
                AddCylinderLayer_and_Material(&brl, "TOB4", Kdtsurfaces, 66, 24);
                AddExtraLayer("ExtraTOBMatLayer5", true, &brl, GenerateTranslation(0, 0, 0), std::make_shared<Acts::CylinderVolumeBounds>(1000*Acts::UnitConstants::mm, 1050*Acts::UnitConstants::mm, 1000*Acts::UnitConstants::mm));
                AddCylinderLayer_and_Material(&brl, "TOB5", Kdtsurfaces, 74, 24);

                //makeLayer("TOB5", 74,10);
              });
        });
        
        det.addCylinderContainer("PositiveTEC", Acts::AxisDirection::AxisZ, [&](auto& ec) {
            ec.setAttachmentStrategy(Acts::VolumeAttachmentStrategy::Gap);
              //.setResizeStrategy(Acts::VolumeResizeStrategy::Gap);

              auto makeLayer = [&](const Acts::Transform3& trf, auto& layer, std::string disk_name, double Bin_Phi, double Bin_R) {
                std::vector<std::shared_ptr<Acts::Surface>> surfaces_disk = SelectActiveSurfaces_PhaseI(Kdtsurfaces, disk_name);
                // Binning:
                makeBinning(layer, Acts::SurfaceArrayNavigationPolicy::LayerType::Disc, Bin_Phi, Bin_R);

                layer.setSurfaces(surfaces_disk)
                    .setLayerType(Acts::Experimental::LayerBlueprintNode::LayerType::Disc)
                    .setEnvelope(Acts::ExtentEnvelope{{
                        .z = {5*Acts::UnitConstants::mm, 5*Acts::UnitConstants::mm},
                        .r = {1*Acts::UnitConstants::mm, 1*Acts::UnitConstants::mm},
                    }})
                    .setTransform(base)
                    .setUseCenterOfGravity(false, false, true); // To fix the transaltion on x-y
              };

              AddDiskLayer_and_Material(&ec, "PosTEC0", makeLayer, 1320*Acts::UnitConstants::mm, 80, 4);
              AddDiskLayer_and_Material(&ec, "PosTEC1", makeLayer, 1463*Acts::UnitConstants::mm, 80, 5);
              AddDiskLayer_and_Material(&ec, "PosTEC2", makeLayer, 1603*Acts::UnitConstants::mm, 80, 5);
              AddDiskLayer_and_Material(&ec, "PosTEC3", makeLayer, 1744*Acts::UnitConstants::mm, 80, 6);
              AddDiskLayer_and_Material(&ec, "PosTEC4", makeLayer, 1883*Acts::UnitConstants::mm, 80, 6);
              AddDiskLayer_and_Material(&ec, "PosTEC5", makeLayer, 2058*Acts::UnitConstants::mm, 80, 6);
              AddExtraLayer("ExtraECMatLater1", false, &ec, GenerateTranslation(0, 0, 2150*Acts::UnitConstants::mm), std::make_shared<Acts::CylinderVolumeBounds>(10*Acts::UnitConstants::mm, 1200*Acts::UnitConstants::mm, 10*Acts::UnitConstants::mm));
              AddDiskLayer_and_Material(&ec, "PosTEC6", makeLayer, 2248*Acts::UnitConstants::mm, 80, 7);
              AddExtraLayer("ExtraECMatLater2", false, &ec, GenerateTranslation(0, 0, 2350*Acts::UnitConstants::mm), std::make_shared<Acts::CylinderVolumeBounds>(10*Acts::UnitConstants::mm, 1200*Acts::UnitConstants::mm, 10*Acts::UnitConstants::mm));
              AddDiskLayer_and_Material(&ec, "PosTEC7", makeLayer, 2454*Acts::UnitConstants::mm, 80, 7);
              AddExtraLayer("ExtraECMatLater3", false, &ec, GenerateTranslation(0, 0, 2550*Acts::UnitConstants::mm), std::make_shared<Acts::CylinderVolumeBounds>(10*Acts::UnitConstants::mm, 1200*Acts::UnitConstants::mm, 10*Acts::UnitConstants::mm));

              ec.addLayer("PosTEC8", [&](auto& layer) {
                makeLayer(base * Acts::Translation3{Acts::Vector3{0, 0, 2666*Acts::UnitConstants::mm}}, layer, "PosTEC8", 80, 7);
              });

            });


        AddExtraLayer("extra_ecpos", false, &det, GenerateTranslation(0, 0, 1150*Acts::UnitConstants::mm), std::make_shared<Acts::CylinderVolumeBounds>(10*Acts::UnitConstants::mm, 1200*Acts::UnitConstants::mm, 10*Acts::UnitConstants::mm));
        AddExtraLayer("extra_ecpos2", false, &det, GenerateTranslation(0, 0, 1200*Acts::UnitConstants::mm), std::make_shared<Acts::CylinderVolumeBounds>(10*Acts::UnitConstants::mm, 1200*Acts::UnitConstants::mm, 10*Acts::UnitConstants::mm));
           
      });
 
      CMS.addStaticVolume(base, std::make_shared<Acts::CylinderVolumeBounds>(0*Acts::UnitConstants::mm, 10*Acts::UnitConstants::mm, 3000*Acts::UnitConstants::mm), "BeamPipe");
    });
  });

  // ===== Construct the TrackingGeometry from the blueprint =====
  Acts::GeometryContext gctx;
  auto logger = Acts::getDefaultLogger("UnitTests", Acts::Logging::INFO);
  Acts::Experimental::BlueprintOptions BluePrint_otp;
  std::shared_ptr<Acts::TrackingGeometry> trackingGeometry = std::move(root->construct(BluePrint_otp, gctx, *logger));

  // ===== OPTIONAL: Save the surfaces associate to the detector elements in an SVG file =====
  if(saveSvgfile_) {
    std::cout << ">>> Flag saveSvgfile True <<<" << std::endl;
    std::cout << "Saving tracking geometry blueprint in " << outputSvgFile_ << std::endl;
    actsvg::views::z_r view_zr; 
    auto svg_objs_rz = ActsPlugins::Svg::drawTrackingGeometry(gctx, *trackingGeometry, view_zr, true, true);
    ActsPlugins::Svg::toFile(svg_objs_rz, outputSvgFile_);
  }

  // ===== Decorate the material using a JSON material file =====
  if (mapMaterial_) {
    std::cout << ">>> Flag mapMaterial_ True <<<" << std::endl;
    std::cout << "Mapping material on sensitive surfaces, from " << materialFile_ << std::endl;
    Acts::MaterialMapJsonConverter::Config dec_cfg;
    Acts::JsonMaterialDecorator jsonMatDec(dec_cfg, materialFile_, Acts::Logging::Level::INFO);

    MatSurfaceSelector sel;
    trackingGeometry->apply(sel);
    std::vector<Acts::Surface*> surfVec = sel.surfaces;

    // Loop over the surfaces and decorate them:
    for(auto& surf : surfVec){
      jsonMatDec.decorate(*surf);
    }
  }

  auto result = std::make_shared<TrackingGeometryWithDetEls>();
    result->detElements = DetEl_vector;
    result->trackingGeometry = trackingGeometry;
  
  return result;
}

DEFINE_FWK_EVENTSETUP_MODULE(ACTSTrackingGeometryProducer);