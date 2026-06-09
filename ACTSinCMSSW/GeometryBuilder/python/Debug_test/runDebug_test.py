# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step3 -s RAW2DIGI,L1Reco,RECO,RECOSIM,PAT,NANO,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM+@nanoAODDQM --conditions auto:phase1_2023_realistic --datatier GEN-SIM-RECO,MINIAODSIM,NANOAODSIM,DQMIO -n 100 --eventcontent RECOSIM,MINIAODSIM,NANOEDMAODSIM,DQM --geometry DB:Extended --era Run3_2023 --no_exec --filein file:step2.root --fileout file:step3.root
import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

# from Configuration.Eras.Era_Run3_2023_cff import Run3_2023
# process = cms.Process('RECO',Run3_2023)
from Configuration.Eras.Era_Run3_noMkFit_cff import Run3_noMkFit
process = cms.Process('RECO',Run3_noMkFit, dd4hep)

#########################################
# GENERATOR INPUT FILE                  #
globalPath = '/eos/user/l/ldamenti/DatasetFarm_files/'
# filename = 'step2_Np10_E10GeV_muNeg_HitDebug.root'
# filename = 'step2_Np10k_E10GeV_muNeg.root'
filename = 'step2_Np1k_E5GeV_muNeg.root'
# filename = 'step2_Np1k_E100GeV_muNeg_BarrOnly.root'
#########################################

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.RecoSim_cff')
process.load('PhysicsTools.PatAlgos.slimming.metFilterPaths_cff')
process.load('Configuration.StandardSequences.PATMC_cff')
process.load('PhysicsTools.NanoAOD.nano_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('DQMServices.Core.DQMStoreNonLegacy_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load("Configuration.Geometry.GeometryDD4hepExtended2023Reco_cff")

# Tracker Geometry and Tracker Topology 
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerTopology_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
    #fileNames = cms.untracked.vstring('file:step2.root'),
    fileNames = cms.untracked.vstring('root://eosuser.cern.ch/' + globalPath + filename),
    secondaryFileNames = cms.untracked.vstring()
)

process.source.skipEvents = cms.untracked.uint32(132) # Jump to event n + 1
# process.source.skipEvents = cms.untracked.uint32(354) # Jump to event n + 1

process.options = cms.untracked.PSet(
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    TryToContinue = cms.untracked.vstring(),
    accelerators = cms.untracked.vstring('*'),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    deleteNonConsumedUnscheduledModules = cms.untracked.bool(True),
    dumpOptions = cms.untracked.bool(False),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(
            allowAnyLabel_=cms.required.untracked.uint32
        ),
        numberOfConcurrentIOVs = cms.untracked.uint32(0)
    ),
    fileMode = cms.untracked.string('FULLMERGE'),
    forceEventSetupCacheClearOnNewRun = cms.untracked.bool(False),
    holdsReferencesToDeleteEarly = cms.untracked.VPSet(),
    makeTriggerResults = cms.obsolete.untracked.bool,
    modulesToCallForTryToContinue = cms.untracked.vstring(),
    modulesToIgnoreForDeleteEarly = cms.untracked.vstring(),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(0),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(0),
    numberOfThreads = cms.untracked.uint32(1),
    printDependencies = cms.untracked.bool(False),
    sizeOfStackForThreadsInKB = cms.optional.untracked.uint32,
    throwIfIllegalParameter = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(False)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step3 nevts:100'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.RECOSIMoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string(f'file:step3.root'),
    #fileName = cms.untracked.string('root://eosuser.cern.ch//eos/user/l/ldamenti/runTheMatrix_files/step3.root'),
    outputCommands = process.RECOSIMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

process.MINIAODSIMoutput = cms.OutputModule("PoolOutputModule",
    compressionAlgorithm = cms.untracked.string('LZMA'),
    compressionLevel = cms.untracked.int32(4),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('MINIAODSIM'),
        filterName = cms.untracked.string('')
    ),
    dropMetaData = cms.untracked.string('ALL'),
    eventAutoFlushCompressedSize = cms.untracked.int32(-900),
    fastCloning = cms.untracked.bool(False),
    fileName = cms.untracked.string(f'file:step3_inMINIAODSIM.root'),
    outputCommands = process.MINIAODSIMEventContent.outputCommands,
    overrideBranchesSplitLevel = cms.untracked.VPSet(
        cms.untracked.PSet(
            branch = cms.untracked.string('patPackedCandidates_packedPFCandidates__*'),
            splitLevel = cms.untracked.int32(99)
        ),
        cms.untracked.PSet(
            branch = cms.untracked.string('recoGenParticles_prunedGenParticles__*'),
            splitLevel = cms.untracked.int32(99)
        ),
        cms.untracked.PSet(
            branch = cms.untracked.string('patTriggerObjectStandAlones_slimmedPatTrigger__*'),
            splitLevel = cms.untracked.int32(99)
        ),
        cms.untracked.PSet(
            branch = cms.untracked.string('patPackedGenParticles_packedGenParticles__*'),
            splitLevel = cms.untracked.int32(99)
        ),
        cms.untracked.PSet(
            branch = cms.untracked.string('patJets_slimmedJets__*'),
            splitLevel = cms.untracked.int32(99)
        ),
        cms.untracked.PSet(
            branch = cms.untracked.string('recoVertexs_offlineSlimmedPrimaryVertices__*'),
            splitLevel = cms.untracked.int32(99)
        ),
        cms.untracked.PSet(
            branch = cms.untracked.string('recoVertexs_offlineSlimmedPrimaryVerticesWithBS__*'),
            splitLevel = cms.untracked.int32(99)
        ),
        cms.untracked.PSet(
            branch = cms.untracked.string('recoCaloClusters_reducedEgamma_reducedESClusters_*'),
            splitLevel = cms.untracked.int32(99)
        ),
        cms.untracked.PSet(
            branch = cms.untracked.string('EcalRecHitsSorted_reducedEgamma_reducedEBRecHits_*'),
            splitLevel = cms.untracked.int32(99)
        ),
        cms.untracked.PSet(
            branch = cms.untracked.string('EcalRecHitsSorted_reducedEgamma_reducedEERecHits_*'),
            splitLevel = cms.untracked.int32(99)
        ),
        cms.untracked.PSet(
            branch = cms.untracked.string('recoGenJets_slimmedGenJets__*'),
            splitLevel = cms.untracked.int32(99)
        ),
        cms.untracked.PSet(
            branch = cms.untracked.string('patJets_slimmedJetsPuppi__*'),
            splitLevel = cms.untracked.int32(99)
        ),
        cms.untracked.PSet(
            branch = cms.untracked.string('EcalRecHitsSorted_reducedEgamma_reducedESRecHits_*'),
            splitLevel = cms.untracked.int32(99)
        )
    ),
    overrideInputFileSplitLevels = cms.untracked.bool(True),
    splitLevel = cms.untracked.int32(0)
)

process.NANOEDMAODSIMoutput = cms.OutputModule("PoolOutputModule",
    compressionAlgorithm = cms.untracked.string('LZMA'),
    compressionLevel = cms.untracked.int32(9),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('NANOAODSIM'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string(f'file:step3_inNANOEDMAODSIM.root'),
    outputCommands = process.NANOAODSIMEventContent.outputCommands
)

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string(f'file:step3_inDQM.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
process.mix.playback = True
process.mix.digitizers = cms.PSet()
for a in process.aliases: delattr(process, a)
process.RandomNumberGeneratorService.restoreStateLabel=cms.untracked.string("randomEngineStateProducer")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2023_realistic', '')
# process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2023_design', '') # TEST

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)
process.recosim_step = cms.Path(process.recosim)
process.Flag_BadChargedCandidateFilter = cms.Path(process.BadChargedCandidateFilter)
process.Flag_BadChargedCandidateSummer16Filter = cms.Path(process.BadChargedCandidateSummer16Filter)
process.Flag_BadPFMuonDzFilter = cms.Path(process.BadPFMuonDzFilter)
process.Flag_BadPFMuonFilter = cms.Path(process.BadPFMuonFilter)
process.Flag_BadPFMuonSummer16Filter = cms.Path(process.BadPFMuonSummer16Filter)
process.Flag_CSCTightHalo2015Filter = cms.Path(process.CSCTightHalo2015Filter)
process.Flag_CSCTightHaloFilter = cms.Path(process.CSCTightHaloFilter)
process.Flag_CSCTightHaloTrkMuUnvetoFilter = cms.Path(process.CSCTightHaloTrkMuUnvetoFilter)
process.Flag_EcalDeadCellBoundaryEnergyFilter = cms.Path(process.EcalDeadCellBoundaryEnergyFilter)
process.Flag_EcalDeadCellTriggerPrimitiveFilter = cms.Path(process.EcalDeadCellTriggerPrimitiveFilter)
process.Flag_HBHENoiseFilter = cms.Path(process.HBHENoiseFilterResultProducer+process.HBHENoiseFilter)
process.Flag_HBHENoiseIsoFilter = cms.Path(process.HBHENoiseFilterResultProducer+process.HBHENoiseIsoFilter)
process.Flag_HcalStripHaloFilter = cms.Path(process.HcalStripHaloFilter)
process.Flag_chargedHadronTrackResolutionFilter = cms.Path(process.chargedHadronTrackResolutionFilter)
process.Flag_ecalBadCalibFilter = cms.Path(process.ecalBadCalibFilter)
process.Flag_ecalLaserCorrFilter = cms.Path(process.ecalLaserCorrFilter)
process.Flag_eeBadScFilter = cms.Path(process.eeBadScFilter)
process.Flag_globalSuperTightHalo2016Filter = cms.Path(process.globalSuperTightHalo2016Filter)
process.Flag_globalTightHalo2016Filter = cms.Path(process.globalTightHalo2016Filter)
process.Flag_goodVertices = cms.Path(process.primaryVertexFilter)
process.Flag_hcalLaserEventFilter = cms.Path(process.hcalLaserEventFilter)
process.Flag_hfNoisyHitsFilter = cms.Path(process.hfNoisyHitsFilter)
process.Flag_muonBadTrackFilter = cms.Path(process.muonBadTrackFilter)
process.Flag_trackingFailureFilter = cms.Path(process.goodVertices+process.trackingFailureFilter)
process.Flag_trkPOGFilters = cms.Path(process.trkPOGFilters)
process.Flag_trkPOG_logErrorTooManyClusters = cms.Path(~process.logErrorTooManyClusters)
process.Flag_trkPOG_manystripclus53X = cms.Path(~process.manystripclus53X)
process.Flag_trkPOG_toomanystripclus53X = cms.Path(~process.toomanystripclus53X)
process.nanoAOD_step = cms.Path(process.nanoSequenceMC)
process.prevalidation_step = cms.Path(process.prevalidationNoHLT)
process.prevalidation_step1 = cms.Path(process.prevalidationMiniAOD)
process.validation_step = cms.EndPath(process.validationNoHLT)
process.validation_step1 = cms.EndPath(process.validationMiniAOD)
process.dqmoffline_step = cms.EndPath(process.DQMOfflineFakeHLT)
process.dqmoffline_1_step = cms.EndPath(process.DQMOfflineMiniAOD)
process.dqmoffline_2_step = cms.EndPath(process.DQMOfflineNanoAOD)
process.dqmofflineOnPAT_step = cms.EndPath(process.PostDQMOffline)
process.dqmofflineOnPAT_1_step = cms.EndPath(process.PostDQMOfflineMiniAOD)
process.dqmofflineOnPAT_2_step = cms.EndPath(process.PostDQMOffline)
process.RECOSIMoutput_step = cms.EndPath(process.RECOSIMoutput)
process.MINIAODSIMoutput_step = cms.EndPath(process.MINIAODSIMoutput)
process.NANOEDMAODSIMoutput_step = cms.EndPath(process.NANOEDMAODSIMoutput)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# ===== Construct the ACTS Tracking Geometry =====
process.trackinGeoProducer = cms.ESProducer("ACTSTrackingGeometryProducer", 
    # Options to save the detector elements in an OBJ file
    saveObjfile    = cms.untracked.bool(False),
    outputObjFile  = cms.untracked.string("checkGeo.obj"),
    rangeZ         = cms.untracked.vdouble(-3000, 3000),  # Min, Max (mm)
    rangeR         = cms.untracked.vdouble(0, 1300),      # Min, Max (mm) 
    # Options to save the Tracker blueprint on an SVG file
    saveSvgfile    = cms.untracked.bool(False),
    outputSvgFile  = cms.untracked.string("testSvg_output.svg"),
    # Option to map the material from a JSON file
    mapMaterial    = cms.untracked.bool(True),
    MaterialMaps   = cms.untracked.string("../MaterialMaps_MatStudy.json")
)

# ===== Call the constructor of the Tracking Geometry =====
process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('ACTSTrackerGeometryRecord'),
        data = cms.vstring('Acts::TrackingGeometry'),
    )),
    verbose = cms.untracked.bool(True))


# ===== Perform the refit using ACTS Producer =====
process.tracksACTSRefit = cms.EDProducer("ACTSRefitTracksProducerDEBUG",
    trackLabels = cms.VInputTag("generalTracks"),
    trackAssociator = cms.untracked.InputTag("quickTrackAssociatorByHits"),  
    trackingParticles = cms.InputTag("mix", "MergedTrackTruth"),

    # TrackingParticleSelector parameters (Disable all the filters)
    ptMinTP = cms.double(0.0),            
    ptMaxTP = cms.double(1e10),          
    minRapidityTP = cms.double(-1e10),    
    maxRapidityTP = cms.double(1e10),     
    tipTP = cms.double(1e10),             
    lipTP = cms.double(1e10),             
    minHitTP = cms.int32(0),              
    signalOnlyTP = cms.bool(False),
    intimeOnlyTP = cms.bool(False),
    chargedOnlyTP = cms.bool(False),
    stableOnlyTP = cms.bool(False),
    pdgIdTP = cms.vint32(),               
    invertRapidityCutTP = cms.bool(False),
    minPhiTP = cms.double(-1e10),        
    maxPhiTP = cms.double(1e10)
)

# ===== Create the output file to give to the DQM Step =====
# I) Save General Track Parameters (i.e. QoverP, Eta, Phi, Pt etc)
process.ACTSTrackingMonitor = process.TrackMon.clone(
    allTrackProducer = cms.InputTag("tracksACTSRefit", "recoTracksCollACTS"),
    TrackProducer    = cms.InputTag("tracksACTSRefit", "recoTracksCollACTS"),
    FolderName       = cms.string('Tracking/ACTSTrackParameters'),
    doGeneralPropertiesPlots = cms.bool(True)
)
# II) Hit Residuals and Pulls (Strips)
process.ACTSMonitorTrackResiduals = process.MonitorTrackResiduals.clone(
    Tracks = cms.InputTag("tracksACTSRefit", "recoTracksCollACTS"),
    TopFolderName = cms.string("ACTSStripResiduals"),
    OutputMEsInRootFile = cms.bool(True),
    trajectoryInput = cms.string('')
)
# III) Hit Residuals and Pulls (Pixel)
process.ACTSSiPixelMonitorTrackResiduals = process.SiPixelMonitorTrackResiduals.clone(
    Tracks = cms.InputTag("tracksACTSRefit", "recoTracksCollACTS"),
    TopFolderName = cms.string("ACTSPixelResiduals"),
    OutputMEsInRootFile = cms.bool(True),
    trajectoryInput = cms.string('')
)
# IV) Track Parameters Residuals and Pulls
from SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi import quickTrackAssociatorByHits
# For ACTS reFitted Tracks:
process.ACTSptAssociation = quickTrackAssociatorByHits.clone(
    label_tp = cms.InputTag("mix","MergedTrackTruth"),
    label_tr = cms.InputTag("tracksACTSRefit", "recoTracksCollACTS")
)
process.ACTSmutliTrackValidator = process.multiTrackValidator.clone(
    associators = cms.untracked.VInputTag("ACTSptAssociation"),
    UseAssociators = cms.bool(True),
    dirName = cms.string("Tracking/ACTSmultiTrackValidator"),
    label = cms.VInputTag(cms.InputTag("tracksACTSRefit", "recoTracksCollACTS")),
    doSummaryPlots = cms.untracked.bool(True),
    doSimPlots = cms.untracked.bool(True),
    doSimTrackPlots = cms.untracked.bool(True),
    doRecoTrackPlots = cms.untracked.bool(True),
    trackCollectionForDrCalculation = cms.InputTag("tracksACTSRefit", "recoTracksCollACTS"),
)
# For CMSSW reFitted Tracks
process.CMSSWptAssociation = quickTrackAssociatorByHits.clone(
    label_tp = cms.InputTag("mix","MergedTrackTruth"),
    label_tr = cms.InputTag("tracksACTSRefit", "recoTracksCollCMSSW")
)
process.CMSSWmutliTrackValidator = process.multiTrackValidator.clone(
    associators = cms.untracked.VInputTag("CMSSWptAssociation"),
    UseAssociators = cms.bool(True),
    dirName = cms.string("Tracking/CMSSWmultiTrackValidator"),
    label = cms.VInputTag(cms.InputTag("tracksACTSRefit", "recoTracksCollCMSSW")),
    doSummaryPlots = cms.untracked.bool(True),
    doSimPlots = cms.untracked.bool(True),
    trackCollectionForDrCalculation = cms.InputTag("tracksACTSRefit", "recoTracksCollCMSSW"),
)

# ====== Sequences ======
process.p = cms.Sequence(
    process.MeasurementTrackerEvent *
    process.tpClusterProducer *
    process.quickTrackAssociatorByHits *
    process.tracksACTSRefit
)
process.load("SimTracker.TrackerHitAssociation.tpClusterProducer_cfi")
process.ACTSval = cms.Sequence(
    process.tpClusterProducer *
    process.ACTSptAssociation *
    process.ACTSmutliTrackValidator
)
process.CMSSWval = cms.Sequence(
    process.tpClusterProducer *
    process.CMSSWptAssociation *
    process.CMSSWmutliTrackValidator
)

# ====== Paths ======
process.Myrefit_step = cms.Path(process.p)
process.MyTrackingDQMPath = cms.Path(process.ACTSTrackingMonitor)
process.MyResidualPath = cms.Path(process.ACTSMonitorTrackResiduals)
process.MyResidualSiPath = cms.Path(process.ACTSSiPixelMonitorTrackResiduals)
process.ActsValidationPath = cms.Path(process.ACTSval)
process.CmsswValidationPath = cms.Path(process.CMSSWval)

# ====== Schedule definition ======
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,
                                process.reconstruction_step,process.recosim_step,
                                process.Flag_HBHENoiseFilter,process.Flag_HBHENoiseIsoFilter,
                                process.Flag_CSCTightHaloFilter,process.Flag_CSCTightHaloTrkMuUnvetoFilter,
                                process.Flag_CSCTightHalo2015Filter,process.Flag_globalTightHalo2016Filter,
                                process.Flag_globalSuperTightHalo2016Filter,process.Flag_HcalStripHaloFilter,
                                process.Flag_hcalLaserEventFilter,process.Flag_EcalDeadCellTriggerPrimitiveFilter,
                                process.Flag_EcalDeadCellBoundaryEnergyFilter,process.Flag_ecalBadCalibFilter,
                                process.Flag_goodVertices,process.Flag_eeBadScFilter,process.Flag_ecalLaserCorrFilter,
                                process.Flag_trkPOGFilters,process.Flag_chargedHadronTrackResolutionFilter,
                                process.Flag_muonBadTrackFilter,process.Flag_BadChargedCandidateFilter,
                                process.Flag_BadPFMuonFilter,process.Flag_BadPFMuonDzFilter,
                                process.Flag_hfNoisyHitsFilter,process.Flag_BadChargedCandidateSummer16Filter,
                                process.Flag_BadPFMuonSummer16Filter,process.Flag_trkPOG_manystripclus53X,
                                process.Flag_trkPOG_toomanystripclus53X,process.Flag_trkPOG_logErrorTooManyClusters,
                                process.nanoAOD_step,process.prevalidation_step,process.prevalidation_step1,
                                process.validation_step,process.validation_step1,process.dqmoffline_step,
                                process.dqmoffline_1_step,process.dqmoffline_2_step,process.dqmofflineOnPAT_step,
                                process.dqmofflineOnPAT_1_step,process.dqmofflineOnPAT_2_step,process.RECOSIMoutput_step,
                                process.MINIAODSIMoutput_step,process.NANOEDMAODSIMoutput_step,
                                process.DQMoutput_step,
                                process.Myrefit_step, process.MyTrackingDQMPath, 
                                process.MyResidualPath, process.MyResidualSiPath, 
                                process.ActsValidationPath, process.CmsswValidationPath)
process.schedule.associate(process.patTask)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# customisation of the process.

# Automatic addition of the customisation function from PhysicsTools.NanoAOD.nano_cff
from PhysicsTools.NanoAOD.nano_cff import nanoAOD_customizeCommon 

#call to customisation function nanoAOD_customizeCommon imported from PhysicsTools.NanoAOD.nano_cff
process = nanoAOD_customizeCommon(process)

# Automatic addition of the customisation function from SimGeneral.MixingModule.fullMixCustomize_cff
from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn 

#call to customisation function setCrossingFrameOn imported from SimGeneral.MixingModule.fullMixCustomize_cff
process = setCrossingFrameOn(process)

# End of customisation functions

# customisation of the process.

# Automatic addition of the customisation function from PhysicsTools.PatAlgos.slimming.miniAOD_tools
from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeAllMC 

#call to customisation function miniAOD_customizeAllMC imported from PhysicsTools.PatAlgos.slimming.miniAOD_tools
rocess = miniAOD_customizeAllMC(process)

# End of customisation functions

# Customisation from command line

#Have logErrorHarvester wait for the same EDProducers to finish as those providing data for the OutputModule
from FWCore.Modules.logErrorHarvester_cff import customiseLogErrorHarvesterUsingOutputCommands
process = customiseLogErrorHarvesterUsingOutputCommands(process)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
