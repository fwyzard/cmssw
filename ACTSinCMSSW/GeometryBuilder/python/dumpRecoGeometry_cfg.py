import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

process = cms.Process("DumpGeometry", dd4hep)

# Geometry: DD4hep Extended2023
process.load("Configuration.Geometry.GeometryDD4hepExtended2023Reco_cff") 
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.ProcessModifiers.dd4hep_cff")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, "auto:phase1_2023_realistic", "")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(1))

# Analyzer
process.dump = cms.EDAnalyzer("TrackerGeometryDumper",
    outputFile = cms.untracked.string("tracker_geometry_Ph1_reco.json")
)

process.p = cms.Path(process.dump)

