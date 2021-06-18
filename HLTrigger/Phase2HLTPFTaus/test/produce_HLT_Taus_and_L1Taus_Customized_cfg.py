# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step3 --geometry Extended2026D49 --era Phase2C9 --conditions 111X_mcRun4_realistic_T15_v4 --processName RECO2 --step RAW2DIGI,RECO --eventcontent RECO --datatier RECO --filein /store/mc/Phase2HLTTDRSummer20ReRECOMiniAOD/QCD_Pt-15to3000_TuneCP5_Flat_14TeV-pythia8/FEVT/PU200_castor_111X_mcRun4_realistic_T15_v1-v1/100000/DA18C0FC-1189-D64B-B3B6-44F3F96F1840.root --mc --nThreads 4 --nStreams 4 --no_exec -n 10 --customise SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000,Configuration/DataProcessing/Utils.addMonitoring --customise JMETriggerAnalysis/Common/customizeHLTForPhase2.customise_hltPhase2_scheduleJMETriggers_TRKv06p1_TICL --customise_commands process.prune()\n --python_filename hltPhase2_TRKv06p1_TICL_cfg.py
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C9_cff import Phase2C9

process = cms.Process('RECO2',Phase2C9)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

#--------------------------------------------------------------------------------
# CV: switch between offline-like and Pixel vertices

vertices = "OfflineVertices"
#vertices = "OnlineVertices"
#vertices = "OnlineVerticesTrimmed"

srcVertices = None
srcBeamSpot = None
if vertices == "OfflineVertices":
    srcVertices = 'offlinePrimaryVertices'
    srcBeamSpot = 'offlineBeamSpot'
elif vertices == "OnlineVertices":
    srcVertices = 'hltPhase2PixelVertices'
    srcBeamSpot = 'hltOnlineBeamSpot'
elif vertices == "OnlineVerticesTrimmed":
    srcVertices = 'hltPhase2TrimmedPixelVertices'
    srcBeamSpot = 'hltOnlineBeamSpot'
else:
    raise ValueError("Invalid configuration parameter vertices = '%s' !!" % vertices)

# CV: enable/disable L1 emulator
#runL1Emulator = False
runL1Emulator = True
#--------------------------------------------------------------------------------

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/mc/Phase2HLTTDRSummer20ReRECOMiniAOD/VBFHToTauTau_M125_14TeV_powheg_pythia8_correctedGridpack_tuneCP5/FEVT/PU200_111X_mcRun4_realistic_T15_v1-v1/120000/084C8B72-BC64-DE46-801F-D971D5A34F62.root'
    ),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
    FailPath = cms.untracked.vstring(),
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    SkipEvent = cms.untracked.vstring(),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(

        ),
        numberOfConcurrentIOVs = cms.untracked.uint32(1)
    ),
    fileMode = cms.untracked.string('FULLMERGE'),
    forceEventSetupCacheClearOnNewRun = cms.untracked.bool(False),
    makeTriggerResults = cms.obsolete.untracked.bool,
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1),
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
    annotation = cms.untracked.string('step3 nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.RECOoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RECO'),
        filterName = cms.untracked.string('')
    ),
    fastCloning = cms.untracked.bool(False),
    fileName = cms.untracked.string('NTuple_produce_HLT_Taus_and_L1Taus_Customized.root'),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_ak4GenJets*_*_*',                   ## PRESENT ONLY IN RAW
        'keep *_hltGtStage2Digis*_*_*',             ## PRESENT ONLY IN RAW
        'keep *_hltTriggerSummaryRAW*_*_*',         ## PRESENT ONLY IN RAW
        'keep *_ak4PFJetsCorrected*_*_*',           ## PRESENT ONLY IN MINIAOD/RECO
        'keep *_hlt*Tau*_*_*',                      ## PRODUCED BY addHLTPFTaus FUNCTION BELOW
        'keep *_particleFlowTmp_*_*',               ## KEEP REFERENCE TO reco::PFCandidate COLLECTION GIVEN AS INPUT TO addHLTPFTaus FUNCTION
        'keep *_muons1stStep_*_*',                  ## KEEP REFERENCE TO reco::Track COLLECTIONS FOR ALL TYPES OF MUONS USED AS INPUT TO PARTICLE-FLOW ALGORITHM
        'keep *_globalMuons_*_*',                   ## KEEP REFERENCE TO reco::Track COLLECTIONS FOR ALL TYPES OF MUONS USED AS INPUT TO PARTICLE-FLOW ALGORITHM
        'keep *_standAloneMuons_*_*',               ## KEEP REFERENCE TO reco::Track COLLECTIONS FOR ALL TYPES OF MUONS USED AS INPUT TO PARTICLE-FLOW ALGORITHM
        'keep *_tevMuons_*_*',                      ## KEEP REFERENCE TO reco::Track COLLECTIONS FOR ALL TYPES OF MUONS USED AS INPUT TO PARTICLE-FLOW ALGORITHM
        'keep *_electronGsfTracks_*_*',             ## KEEP REFERENCE TO reco::GsfTrack COLLECTION FOR ELECTRONS USED AS INPUT TO PARTICLE-FLOW ALGORITHM
        'keep *_generalTracks_*_*',                 ## KEEP REFERENCE TO reco::Track COLLECTION GIVEN AS INPUT TO addHLTPFTaus FUNCTION
        'keep *_offlinePrimaryVertices_*_*',        ## KEEP REFERENCE TO reco::Vertex COLLECTION GIVEN AS INPUT TO addHLTPFTaus FUNCTION 
        'keep *_hltPhase2PixelVertices_*_*',        ## PRODUCED BELOW
        'keep *_hltPhase2TrimmedPixelVertices_*_*', ## PRODUCED BELOW
        'keep *_hltKT6PFJets_*_*',                  ## PRODUCED BELOW
        'keep *_hltPFMET*_*_*',                     ## PRODUCED BELOW
        'keep *_hltPuppiMET*_*_*',                  ## PRODUCED BELOW
        'keep *_hltPackedPFCandidates*_*_*',        ## PRODUCED BELOW
        'keep *_prunedGenParticles_*_*',            ## PRESENT ONLY IN MINIAOD/RECO
        'keep *_ak4GenJets_*_*',                    ## PRESENT ONLY IN MINIAOD/RECO
        'keep *_ak8GenJets_*_*',                    ## PRESENT ONLY IN MINIAOD/RECO
        'keep *_slimmedGenJets__*',                 ## PRESENT ONLY IN MINIAOD/RECO
        'keep *_slimmedTaus_*_*',                   ## PRESENT ONLY IN MINIAOD/RECO
        'keep *_slimmedJets_*_*',                   ## PRESENT ONLY IN MINIAOD/RECO
        'keep *_packedPFCandidates_*_*',            ## PRESENT ONLY IN MINIAOD/RECO
        'keep *_slimmedAddPileupInfo_*_*',          ## PRESENT ONLY IN MINIAOD/RECO
        'keep *_addPileupInfo_*_*',                 ## PRESENT ONLY IN RAW
        'keep *_offlineSlimmedPrimaryVertices_*_*', ## PRESENT ONLY IN MINIAOD/RECO
        'keep *_generatorSmeared_*_*',              ## CV: ALLOWS TO PRODUCE FULL COLLECTION OF genParticles FOR DEBUGGING PURPOSES 
        'keep *_generator_*_*',                     ## CV: NEEDED TO MAKE PTHAT PLOTS FOR QCD MULTIJET MC SAMPLES
        'keep *_HPSPFTauProducer*PF_*_*',         ## ADDED BY L1 EMULATOR
        'keep *_l1pfCandidates_PF_*',               ## ADDED BY L1 EMULATOR
        'keep *_l1pfProducer*_z0_*',                ## ADDED BY L1 EMULATOR
        'keep *_pfTracksFromL1Tracks*_*_*',         ## ADDED BY L1 EMULATOR
        'keep *_pfClustersFrom*_*_*',               ## ADDED BY L1 EMULATOR
        'keep *_TTTracksFromTracklet_*_*',          ## ADDED BY L1 EMULATOR
        'keep *_VertexProducer_*_*',                ## ADDED BY L1 EMULATOR
        'keep *_ak4PFL1PF_*_*',                     ## ADDED BY L1 EMULATOR
        'keep *_ak4PFL1PFCorrected_*_*',            ## ADDED BY L1 EMULATOR
        'keep *_kt6L1PFJetsPF_rho_*',               ## ADDED BY L1 EMULATOR
        'keep *_kt6L1PFJetsNeutralsPF_rho_*',       ## ADDED BY L1 EMULATOR
        'keep *_l1pfCandidates_PF_*',               ## ADDED BY L1 EMULATOR
        'keep *_l1pfCandidates_Puppi_*',            ## ADDED BY L1 EMULATOR
        'keep *_L1TkPrimaryVertex_*_*',             ## ADDED BY L1 EMULATOR
        'keep *_HPSPFTauProducer*PF_*_*',         ## ADDED BY L1 EMULATOR
        'keep *_HPSPFTauProducer*Puppi_*_*',      ## ADDED BY L1 EMULATOR
        'keep *_*BeamSpot*_*_*',                    ## Need the beamspot
        "keep *_*_L1PFTausNN_*",                    ## Save NNPUPPI Tau
    )

)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '111X_mcRun4_realistic_T15_v4', '')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.reconstruction_step = cms.Path(process.reconstruction)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RECOoutput_step = cms.EndPath(process.RECOoutput)

# customisation of the process.

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.aging
from SLHCUpgradeSimulations.Configuration.aging import customise_aging_1000

#call to customisation function customise_aging_1000 imported from SLHCUpgradeSimulations.Configuration.aging
process = customise_aging_1000(process)

# Automatic addition of the customisation function from Configuration.DataProcessing.Utils
from Configuration.DataProcessing.Utils import addMonitoring

#call to customisation function addMonitoring imported from Configuration.DataProcessing.Utils
process = addMonitoring(process)

# Automatic addition of the customisation function from JMETriggerAnalysis.Common.customizeHLTForPhase2
from JMETriggerAnalysis.Common.customizeHLTForPhase2 import customise_hltPhase2_scheduleJMETriggers_TRKv06p1_TICL

#call to customisation function customise_hltPhase2_scheduleJMETriggers_TRKv06p1_TICL imported from JMETriggerAnalysis.Common.customizeHLTForPhase2
process = customise_hltPhase2_scheduleJMETriggers_TRKv06p1_TICL(process)

# End of customisation functions
process.source.inputCommands=cms.untracked.vstring('keep *_*_*_*')

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.endjob_step,process.RECOoutput_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

#Setup FWK for multithreaded
process.options.numberOfThreads=cms.untracked.uint32(4)
process.options.numberOfStreams=cms.untracked.uint32(4)
process.options.numberOfConcurrentLuminosityBlocks=cms.untracked.uint32(1)

#-------------------------------------------------------------------------------- 
# CV: run L1 trigger emulator and produce L1 HPS PF Tau objects
if runL1Emulator:
    process.load('L1Trigger.Phase2L1Taus.l1emulator_cff')
    process.l1emulatorSequence = cms.Sequence(process.l1emulator)

    # SB: produce L1 HPS PF Tau objects                                                                                                                                                                                                                                         
    from L1Trigger.Phase2L1Taus.HPSPFTauProducerPF_cfi import HPSPFTauProducerPF
    from L1Trigger.Phase2L1Taus.HPSPFTauProducerPuppi_cfi import HPSPFTauProducerPuppi
    for useStrips in [ True, False ]:
        moduleNameBase = "HPSPFTauProducer"
        if useStrips:
            moduleNameBase += "WithStrips"
        else:
            moduleNameBase += "WithoutStrips"

        moduleNamePF = moduleNameBase + "PF"
        modulePF = HPSPFTauProducerPF.clone(
            useStrips = cms.bool(useStrips),
            applyPreselection = cms.bool(False),
            debug = cms.untracked.bool(False)
        )
        setattr(process, moduleNamePF, modulePF)
        process.l1emulatorSequence += getattr(process, moduleNamePF)

    process.reconstruction_step += process.l1emulatorSequence
#-------------------------------------------------------------------------------- 

#--------------------------------------------------------------------------------
# produce L1 PF and NN Taus
if runL1Emulator:
    process.load("L1Trigger.Phase2L1ParticleFlow.L1NNTauProducer_cff")
    process.L1NNTauProducer.L1PFObjects = cms.InputTag("l1pfCandidates","PF")
    process.reconstruction_step += process.L1NNTauProducer

    process.L1NNTauProducerPuppi = process.L1NNTauProducer.clone()
    process.L1NNTauProducerPuppi.L1PFObjects = cms.InputTag("l1pfCandidates","Puppi")
    process.reconstruction_step += process.L1NNTauProducerPuppi

#--------------------------------------------------------------------------------

#from HLTrigger.Phase2HLTPFTaus.particleFlowSuperClusterHGCalFromMultiCl_cfi import particleFlowSuperClusterHGCalFromMultiCl
from HLTrigger.Phase2.HLT_75e33.modules.particleFlowSuperClusterHGCalFromMultiCl_cfi import particleFlowSuperClusterHGCalFromMultiCl
process.particleFlowSuperClusterHGCalFromMultiCl.useRegression = cms.bool(False)
process.reconstruction_step += process.particleFlowSuperClusterHGCalFromMultiCl

#--------------------------------------------------------------------------------
# CV: add HLT tau reconstruction
process.taucustomreco = cms.Sequence()

# run HLT tau reconstruction
from HLTrigger.Phase2HLTPFTaus.tools.addHLTPFTaus import addHLTPFTaus
srcPFCandidates = "particleFlowTmp"
for algorithm in [ "hps", "shrinking-cone" ]:
  for isolation_maxDeltaZOption in [ "primaryVertex", "leadTrack" ]:
    for isolation_minTrackHits in [ 3, 5, 8 ]:

      suffix = "%iHits" % isolation_minTrackHits
      isolation_maxDeltaZ            = None
      isolation_maxDeltaZToLeadTrack = None
      if isolation_maxDeltaZOption == "primaryVertex":
        isolation_maxDeltaZ            =  0.15 # value optimized for offline tau reconstruction at higher pileup expected during LHC Phase-2
        isolation_maxDeltaZToLeadTrack = -1.   # disabled
        suffix += "MaxDeltaZ"
      elif isolation_maxDeltaZOption == "leadTrack":
        isolation_maxDeltaZ            = -1.   # disabled
        isolation_maxDeltaZToLeadTrack =  0.15 # value optimized for offline tau reconstruction at higher pileup expected during LHC Phase-2
        suffix += "MaxDeltaZToLeadTrack"
      else:
        raise ValueError("Invalid parameter isolation_maxDeltaZOption = '%s' !!" % isolation_maxDeltaZOption)
      if srcVertices == "offlinePrimaryVertices":
        suffix += "WithOfflineVertices"
      elif srcVertices == "hltPhase2PixelVertices":
        suffix += "WithOnlineVertices"
      elif srcVertices == "hltPhase2TrimmedPixelVertices":
        suffix += "WithOnlineVerticesTrimmed"
      else:
        raise ValueError("Invalid parameter srcVertices = '%s' !!" % srcVertices)

      pftauSequence = addHLTPFTaus(process, algorithm,
        srcPFCandidates, srcVertices, srcBeamSpot,
        isolation_maxDeltaZ, isolation_maxDeltaZToLeadTrack, isolation_minTrackHits,
        suffix)
      process.taucustomreco += pftauSequence

process.reconstruction_step += process.taucustomreco

hlt_pfTauLabel = 'HpsPFTau'
hlt_srcVertices = 'offlinePrimaryVertices'
suffix = '8HitsMaxDeltaZWithOfflineVertices'
from HLTrigger.Phase2HLTPFTaus.tools.addDeepTauDiscriminator import addDeepTauDiscriminator
hlt_srcPFTaus = 'hltSelected%ss%s' % (hlt_pfTauLabel, suffix)
hlt_srcPFJets = 'hlt%sAK4PFJets%s' % (hlt_pfTauLabel, suffix)
deepTauSequenceName = "hltDeep%sSequence%s" % (hlt_pfTauLabel, suffix)
deepTauSequence = addDeepTauDiscriminator(process, hlt_srcPFTaus, hlt_srcPFJets, hlt_srcVertices,
        hlt_pfTauLabel, suffix, deepTauSequenceName)
process.reconstruction_step += deepTauSequence


# CV: add kt6PFJets for rho computation
from RecoJets.JetProducers.kt6PFJets_cfi import kt6PFJets
process.hltKT6PFJets = kt6PFJets.clone(
    src = cms.InputTag("particleFlowTmp"),
    doRhoFastjet = cms.bool(True)
)
process.reconstruction_step += process.hltKT6PFJets
#--------------------------------------------------------------------------------


#-------------------------------------------------------------------------------- 
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

dump_file = open('dump.py','w')
dump_file.write(process.dumpPython())

process.options.numberOfThreads = cms.untracked.uint32(4)

#--------------------------------------------------------------------------------
