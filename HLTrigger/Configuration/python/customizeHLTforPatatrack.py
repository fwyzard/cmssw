import FWCore.ParameterSet.Config as cms

from HLTrigger.Configuration.common import *

def customiseHLTforTestingDQMGPUvsCPU(process):
    '''Ad-hoc changes to test HLT config containing only DQM_PixelReconstruction_v and DQMGPUvsCPU stream
    '''
    if hasattr(process, 'hltDatasetDQMGPUvsCPU'):
        process.hltDatasetDQMGPUvsCPU.triggerConditions = ['DQM_PixelReconstruction_v*']

    finalPathsToRemove = []
    for fpath in process.finalpaths_():
      if 'DQMGPUvsCPU' not in fpath:
        finalPathsToRemove += [fpath]
    for fpath in finalPathsToRemove:
      process.__delattr__(fpath)

    return process

def customiseHLTforTestingDQMGPUvsCPUPixelOnlyUpToLocal(process):
    '''Ad-hoc changes to test HLT config containing only DQM_PixelReconstruction_v and DQMGPUvsCPU stream
       only up to the Pixel Local Reconstruction
    '''
    process = customiseHLTforTestingDQMGPUvsCPU(process)

    if not hasattr(process, 'HLTDoLocalPixelTask'):
        return process

#     process.HLTDoLocalPixelTask = cms.ConditionalTask(
#         process.hltSiPixelClustersCPU,
#         process.hltSiPixelClustersGPU,
# 
# #        process.hltSiPixelDigiErrorsSoA,
# #        process.hltSiPixelDigiErrorsSoALegacy,
#         process.hltSiPixelDigisSoA,
# 
#         process.hltSiPixelDigisFromSoALegacy,
#         process.hltSiPixelDigisFromSoA,
#         process.hltSiPixelClustersFromSoA,
#         process.hltSiPixelClustersLegacy,
# 
# ###        process.hltSiPixelDigisLegacy,
# ###        process.hltSiPixelDigis,
# ###        process.hltSiPixelClusters,
# ###        process.hltSiPixelClustersCache,
# ###
# ###        process.hltOnlineBeamSpotToGPU,
# ###        process.hltSiPixelRecHitsFromLegacy,
# ###        process.hltSiPixelRecHitsGPU,
# ###        process.hltSiPixelRecHitsFromGPU,
# ###        process.hltSiPixelRecHits,
# ###        process.hltSiPixelRecHitsSoAFromGPU,
# ###        process.hltSiPixelRecHitsSoA
#     )

    process.hltPixelConsumerCPU.eventProducts = [
        'hltSiPixelClustersCPUSerial',
        'hltSiPixelDigiErrorsCPUSerial',
#        'hltSiPixelRecHitsCPUSerial', # leads to exception
    ]

    process.hltPixelConsumerGPU.eventProducts = [
        'hltSiPixelClusters',
        'hltSiPixelDigiErrors',
        'hltSiPixelRecHits',
    ]

    # modify EventContent of DQMGPUvsCPU stream
    if hasattr(process, 'hltOutputDQMGPUvsCPU'):
        process.hltOutputDQMGPUvsCPU.outputCommands = [
          'drop *',
          'keep *Cluster*_hltSiPixelClusters_*_*',
          'keep *Cluster*_hltSiPixelClustersCPUSerial_*_*',
          'keep *_hltSiPixelDigiErrors_*_*',
          'keep *_hltSiPixelDigiErrorsCPUSerial_*_*',
#          'keep *RecHit*_hltSiPixelRecHits_*_*',
#          'keep *RecHit*_hltSiPixelRecHitsCPUSerial_*_*',
        ]

    # empty HLTRecopixelvertexingSequence until we add tracks and vertices
    process.HLTRecopixelvertexingSequence = cms.Sequence()

    # create CPU version of LocalPixelRecoSequence, and add it to HLTDQMPixelReconstruction
    process.HLTDoLocalPixelSequenceCPUSerial = cms.Sequence( process.HLTDoLocalPixelTaskCPUSerial )
    process.HLTDQMPixelReconstruction.insert(0, process.HLTDoLocalPixelSequenceCPUSerial)

    return process

def customiseHLTforAlpakaPixelRecoLocal(process):
    '''Customisation to introduce the Local Pixel Reconstruction in Alpaka
    '''
    process.hltESPSiPixelCablingSoAESProducer = cms.ESProducer('SiPixelCablingSoAESProducer@alpaka',
        ComponentName = cms.string(''),
        CablingMapLabel = cms.string(''),
        UseQualityInfo = cms.bool(False),
        appendToDataLabel = cms.string(''),
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    process.hltESPSiPixelGainCalibrationForHLTSoAESProducer = cms.ESProducer('SiPixelGainCalibrationForHLTSoAESProducer@alpaka',
        appendToDataLabel = cms.string(''),
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    process.hltESPPixelCPEFastParamsESProducerPhase1 = cms.ESProducer('PixelCPEFastParamsESProducerAlpakaPhase1@alpaka',
        ComponentName = cms.string('PixelCPEFast'),
        appendToDataLabel = cms.string(''),
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    ###

    # alpaka EDProducer
    # consumes
    #  - FEDRawDataCollection
    # produces (* optional)
    #  - SiPixelClustersSoA
    #  - SiPixelDigisSoA
    #  - SiPixelDigiErrorsSoA *
    #  - SiPixelFormatterErrors *
    process.hltSiPixelClusters = cms.EDProducer('SiPixelRawToCluster@alpaka',
        isRun2 = cms.bool(False),
        IncludeErrors = cms.bool(True),
        UseQualityInfo = cms.bool(False),
        clusterThreshold_layer1 = cms.int32(4000),
        clusterThreshold_otherLayers = cms.int32(4000),
        InputLabel = cms.InputTag('rawDataCollector'),
        Regions = cms.PSet(
            inputs = cms.optional.VInputTag,
            deltaPhi = cms.optional.vdouble,
            maxZ = cms.optional.vdouble,
            beamSpot = cms.optional.InputTag
        ),
        CablingMapLabel = cms.string(''),
        # autoselect the alpaka backend
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    process.hltSiPixelClustersCPUSerial = process.hltSiPixelClusters.clone(
        alpaka = dict( backend = 'serial_sync' )
    )

    # legacy EDProducer
    # consumes
    #  - SiPixelDigiErrorsHost
    #  - SiPixelFormatterErrors
    # produces
    #  - edm::DetSetVector<SiPixelRawDataError>
    #  - DetIdCollection
    #  - DetIdCollection, "UserErrorModules"
    #  - edmNew::DetSetVector<PixelFEDChannel>
    process.hltSiPixelDigiErrors = cms.EDProducer("SiPixelDigiErrorsFromSoA",
        digiErrorSoASrc = cms.InputTag("hltSiPixelClusters"),
        fmtErrorsSoASrc = cms.InputTag("hltSiPixelClusters"),
        CablingMapLabel = cms.string(''),
        UsePhase1 = cms.bool(True),
        ErrorList = cms.vint32(29),
        UserErrorList = cms.vint32(40)
    )

    process.hltSiPixelDigiErrorsCPUSerial = process.hltSiPixelDigiErrors.clone(
        digiErrorSoASrc = "hltSiPixelClustersCPUSerial",
        fmtErrorsSoASrc = "hltSiPixelClustersCPUSerial",
    )

    # alpaka EDProducer
    # consumes
    #  - reco::BeamSpot
    # produces
    #  - BeamSpotDeviceProduct
    process.hltOnlineBeamSpotDevice = cms.EDProducer("BeamSpotDeviceProducer@alpaka",
        src = cms.InputTag("hltOnlineBeamSpot"),
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    process.hltOnlineBeamSpotDeviceCPUSerial = process.hltOnlineBeamSpotDevice.clone(
        alpaka = dict( backend = 'serial_sync' )
    )

    # alpaka EDProducer
    # consumes
    #  - BeamSpotDeviceProduct
    #  - SiPixelClustersSoA
    #  - SiPixelDigisSoA
    # produces
    #  - TrackingRecHitAlpakaCollection<TrackerTraits>
    process.hltSiPixelRecHits = cms.EDProducer("SiPixelRecHitAlpakaPhase1@alpaka",
        beamSpot = cms.InputTag('hltOnlineBeamSpotDevice'),
        src = cms.InputTag('hltSiPixelClusters'),
        CPE = cms.string('PixelCPEFast'),
        # autoselect the alpaka backend
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string('')
        )
    )

    process.hltSiPixelRecHitsCPUSerial = process.hltSiPixelRecHits.clone(
        beamSpot = 'hltOnlineBeamSpotDeviceCPUSerial',
        src = 'hltSiPixelClustersCPUSerial',
        alpaka = dict( backend = 'serial_sync' )
    )

#    # alpaka EDProducer
#    # consumes
#    #  - TrackingRecHitAlpakaCollection<TrackerTraits>
#    # produces
#    #  - TkSoADevice
#    process.hltPixelTracks = cms.EDProducer("CAHitNtupletAlpakaPhase1@alpaka",
#        onGPU = cms.bool(True),
#        pixelRecHitSrc = cms.InputTag('hltSiPixelRecHits'),
#        ptmin = cms.double(0.89999997615814209),
#        CAThetaCutBarrel = cms.double(0.0020000000949949026),
#        CAThetaCutForward = cms.double(0.0030000000260770321),
#        hardCurvCut = cms.double(0.032840722495894911),
#        dcaCutInnerTriplet = cms.double(0.15000000596046448),
#        dcaCutOuterTriplet = cms.double(0.25),
#        earlyFishbone = cms.bool(True),
#        lateFishbone = cms.bool(False),
#        fillStatistics = cms.bool(False),
#        minHitsPerNtuplet = cms.uint32(3),
#        maxNumberOfDoublets = cms.uint32(524288),
#        minHitsForSharingCut = cms.uint32(10),
#        fitNas4 = cms.bool(False),
#        doClusterCut = cms.bool(True),
#        doZ0Cut = cms.bool(True),
#        doPtCut = cms.bool(True),
#        useRiemannFit = cms.bool(False),
#        doSharedHitCut = cms.bool(True),
#        dupPassThrough = cms.bool(False),
#        useSimpleTripletCleaner = cms.bool(True),
#        idealConditions = cms.bool(False),
#        includeJumpingForwardDoublets = cms.bool(True),
#        trackQualityCuts = cms.PSet(
#            chi2MaxPt = cms.double(10),
#            chi2Coeff = cms.vdouble(0.9, 1.8),
#            chi2Scale = cms.double(8),
#            tripletMinPt = cms.double(0.5),
#            tripletMaxTip = cms.double(0.3),
#            tripletMaxZip = cms.double(12),
#            quadrupletMinPt = cms.double(0.3),
#            quadrupletMaxTip = cms.double(0.5),
#            quadrupletMaxZip = cms.double(12)
#        ),
#        # autoselect the alpaka backend
#        alpaka = cms.untracked.PSet(
#            backend = cms.untracked.string('')
#        )
#    )
#
#    # alpaka EDProducer
#    # consumes
#    #  - TkSoADevice
#    # produces
#    #  - ZVertexDevice
#    process.hltPixelVertices = cms.EDProducer('PixelVertexProducerAlpakaPhase1@alpaka',
#        onGPU = cms.bool(True),
#        oneKernel = cms.bool(True),
#        useDensity = cms.bool(True),
#        useDBSCAN = cms.bool(False),
#        useIterative = cms.bool(False),
#        minT = cms.int32(2),
#        eps = cms.double(0.07),
#        errmax = cms.double(0.01),
#        chi2max = cms.double(9),
#        PtMin = cms.double(0.5),
#        PtMax = cms.double(75),
#        pixelTrackSrc = cms.InputTag('pixelTracksCUDA'),
#        # autoselect the alpaka backend
#        alpaka = cms.untracked.PSet(
#            backend = cms.untracked.string('')
#        )
#    )

    process.HLTDoLocalPixelTask = cms.ConditionalTask(
        process.hltSiPixelClusters,
        process.hltSiPixelDigiErrors,
        process.hltOnlineBeamSpotDevice,
        process.hltSiPixelRecHits,
    )

    process.HLTDoLocalPixelTaskCPUSerial = cms.ConditionalTask(
        process.hltSiPixelClustersCPUSerial,
        process.hltSiPixelDigiErrorsCPUSerial,
        process.hltOnlineBeamSpotDeviceCPUSerial,
        process.hltSiPixelRecHitsCPUSerial,
    )

    return process

def customiseHLTforAlpakaPixelRecoTracking(process):
    '''Customisation to introduce the Pixel-Track Reconstruction in Alpaka
    '''
    return process

def customiseHLTforAlpakaPixelRecoVertexing(process):
    '''Customisation to introduce the Pixel-Vertex Reconstruction in Alpaka
    '''
    return process

def customiseHLTforAlpakaPixelReco(process):
    '''Customisation to introduce the Pixel Local+Track+Vertex Reconstruction in Alpaka
    '''
    process.load('Configuration.StandardSequences.Accelerators_cff')
    process.load('HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi')

    process = customiseHLTforAlpakaPixelRecoLocal(process)
#    process = customiseHLTforAlpakaPixelRecoTracking(process)
#    process = customiseHLTforAlpakaPixelRecoVertexing(process)
    return process

def customizeHLTforPatatrack(process):
    '''Customise HLT configuration introducing latest Patatrack developments
    '''
    process = customiseHLTforAlpakaPixelReco(process)
    return process
