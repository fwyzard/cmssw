import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

process = cms.Process("DumpGeometry", dd4hep)

isPhase2 = False

# Load DD4hep Run4 geometry
if(isPhase2):
    process.load("Configuration.Geometry.GeometryDD4hepExtendedRun4DefaultReco_cff") # PHASE 2 (reco to avoid to add many plugins. Phase 1 is simpler and I can use the non-reco one)
else:
    process.load("Configuration.Geometry.GeometryDD4hepExtended2023_cff")     # PHASE 1


# Tracker Geometry and Tracker Topology 
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerTopology_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cff")

if not isPhase2: process.TrackerAdditionalParametersPerDetESModule = cms.ESProducer("TrackerAdditionalParametersPerDetESModule")

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.ProcessModifiers.dd4hep_cff")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, "auto:phase2_realistic", "") if (isPhase2) else GlobalTag(process.GlobalTag, "auto:phase1_2023_realistic", "") 

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(1))

output_filename = "tracker_geometry_Ph2.json" if (isPhase2) else "tracker_geometry_Ph1_FromReco.json"
analyzer = "TrackerGeometryDumper_Phase2" if (isPhase2) else "TrackerGeometryDumper"

process.dump = cms.EDAnalyzer(analyzer,
    outputFile = cms.untracked.string(output_filename)
)

process.p = cms.Path(process.dump)