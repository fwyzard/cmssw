'''
NOTE: How to use this?
First you need to create the json file. To do so change the name of the output file in the Producer 'createMaterialFile' and 
add ONLY this producer in the Path. 
Then, once the file is created, you can validate it by using the file you just created in the Producer 'trackinGeoProducer' and 
add ONLY the 'validateMappedMat' in the Path.
Once this job is finished, run 'python3 validateMaterial.py' to make the comparison plots
'''

import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

process = cms.Process("DumpGeometry", dd4hep)

process.TrackerRecoGeometryESProducer = cms.ESProducer("TrackerRecoGeometryESProducer",
  usePhase2Stacks = cms.bool(False)
)

# Load DD4hep geometry
process.load("Configuration.Geometry.GeometryDD4hepExtended2023_cff")     # PHASE 1

# Tracker Geometry and Tracker Topology 
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerTopology_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cff")

process.TrackerAdditionalParametersPerDetESModule = cms.ESProducer("TrackerAdditionalParametersPerDetESModule")

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

# process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, "auto:phase1_2023_realistic", "") 

# # ====== Track associator & TP cluster producer ======
# process.load("SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi")
# from SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi import quickTrackAssociatorByHits
# process.trackAssociator = quickTrackAssociatorByHits.clone()

# (
#     Quality_SimToReco = cms.double(0.4),
#     Purity_SimToReco = cms.double(0.4),
#     SimToRecoDenominator = cms.string('sim'),    # calcola la frazione rispetto alla TP
#     Cut_RecoToSim = cms.double(0.4),
# )

# from SimTracker.TrackerHitAssociation.tpClusterProducerDefault_cfi import tpClusterProducerDefault
# process.tpClusterProducer = tpClusterProducerDefault.clone()

# # ====== Cluster position estimators ======
# from RecoLocalTracker.SiPixelRecHits.PixelCPEGeneric_cfi import PixelCPEGenericESProducer
# process.PixelCPEGenericESProducer = PixelCPEGenericESProducer.clone()

# # include to compute the hit position and error given the cluster
# from Configuration.AlCa.GlobalTag import GlobalTag
# process.GlobalTag = GlobalTag(process.GlobalTag, "auto:phase1_2023_realistic", "")
# from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import StripCPEfromTrackAngleESProducer
# process.StripCPEfromTrackAngleESProducer = StripCPEfromTrackAngleESProducer.clone()

# ====== Logging ======
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.FwkReport.reportEvery = 1

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

# # ====== Input ======
# process.source = cms.Source("PoolSource",
#     fileNames = cms.untracked.vstring(
#         #'file:muonGun10GeV_pythia8_Run3dd4hep_DBrealistic.root'
#         'root://eosuser.cern.ch//eos/user/l/ldamenti/1k_100GeVnegMu_WithB_BarrelOnly.root')
# )
# process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

# # ====== MeasurementTrackerEvent ======
# process.MeasurementTrackerEvent = cms.EDProducer("MeasurementTrackerEventProducer",
#     stripClusterProducer = cms.string("siStripClusters"),
#     pixelClusterProducer = cms.string("siPixelClusters"),
# )

# ===== Construct the ACTS Tracking Geometry =====
process.trackinGeoProducer = cms.ESProducer("ACTSTrackingGeometryProducer", 
    # Options to save the detector elements in an OBJ file
    saveObjfile    = cms.untracked.bool(False),
    outputObjFile  = cms.untracked.string("testSlice.obj"),
    rangeZ         = cms.untracked.vdouble(-1000, 1000),  # Min, Max (mm)
    rangeR         = cms.untracked.vdouble(0, 1200),      # Min, Max (mm) 
    # Options to save the Tracker blueprint on an SVG file
    saveSvgfile    = cms.untracked.bool(False),
    outputSvgFile  = cms.untracked.string("testSvg_output.svg"),
    # Option to map the material from a JSON file
    mapMaterial    = cms.untracked.bool(True),
    MaterialMaps   = cms.untracked.string("MaterialMaps_MatStudy.json")
)

process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('ACTSTrackerGeometryRecord'),
        data = cms.vstring('Acts::TrackingGeometry'),
    )),
    verbose = cms.untracked.bool(True))

# ===== Create the Material Maps =====
process.createMaterialFile = cms.EDProducer("JsonMaterialMapsProducer",
    G4InputFile = cms.untracked.string("/eos/user/l/ldamenti/G4MaterialFiles/MaterialFile_ExtendedGeo2023_10000_100_test_TrkOnly.root"),
    OutputFile  = cms.untracked.string("MaterialMaps_MatStudy.json"),
    Nevents = cms.untracked.int32(10000),
)

# ===== Validate the Mapped Material =====
process.validateMappedMat = cms.EDProducer("MappedMaterialValidator", 
    Nevents = cms.untracked.int32(100),
    Ntracks = cms.untracked.int32(100),
)

# ====== Paths ======
# Uncomment to create material file 
# process.p = cms.Path(process.createMaterialFile)
# Uncomment to validate the material map
process.p = cms.Path(process.validateMappedMat)


