import FWCore.ParameterSet.Config as cms

#--------------------------------------------------------------------------------
# CV: config file downloaded from
#       https://github.com/hevjinyarar/CMS_HLT_Phase2_Tracking/blob/master/MC_Tracking_CMSSW_11_1_0_pre3/MC_Tracking_v6_cff.py
#     on 04/06/2020
#--------------------------------------------------------------------------------

############## pixelTracks/Vertices

hltPhase2PSetPvClusterComparerForIT = cms.PSet( 
  track_chi2_max = cms.double( 20.0 ),
  track_pt_max = cms.double( 20.0 ),
  track_prob_min = cms.double( -1.0 ),
  track_pt_min = cms.double( 1.0 )
)


hltPhase2PixelTrackCleanerBySharedHits = cms.ESProducer( "PixelTrackCleanerBySharedHitsESProducer",
  useQuadrupletAlgo = cms.bool( False ),
  ComponentName = cms.string( "hltPhase2PixelTrackCleanerBySharedHits" ),
  appendToDataLabel = cms.string( "" )
)


hltPhase2PixelTrackFilterByKinematics = cms.EDProducer( "PixelTrackFilterByKinematicsProducer",
    nSigmaTipMaxTolerance = cms.double( 0.0 ),
    chi2 = cms.double( 1000.0 ),
    nSigmaInvPtTolerance = cms.double( 0.0 ),
    ptMin = cms.double( 0.9 ), #previous 0.1
    tipMax = cms.double( 1.0 )
)

hltPhase2PixelFitterByHelixProjections = cms.EDProducer( "PixelFitterByHelixProjectionsProducer",
    scaleErrorsForBPix1 = cms.bool( False ),
    scaleFactor = cms.double( 0.65 )
)


hltPhase2PixelTracksHitQuadruplets = cms.EDProducer( "CAHitQuadrupletEDProducer",
    CAHardPtCut = cms.double( 0.0 ),
    SeedComparitorPSet = cms.PSet( 
      clusterShapeHitFilter = cms.string( "ClusterShapeHitFilter" ),
      ComponentName = cms.string( "LowPtClusterShapeSeedComparitor" ),
      clusterShapeCacheSrc = cms.InputTag( "siPixelClusterShapeCache") # pixelVertices
    ),
    extraHitRPhitolerance = cms.double( 0.032 ),
    doublets = cms.InputTag( "hltPhase2PixelTracksHitDoublets" ), 
    fitFastCircle = cms.bool( True ),
    CAThetaCut = cms.double( 0.0012 ), # 0.002 ),
    maxChi2 = cms.PSet( 
      value2 = cms.double( 50.0 ),
      value1 = cms.double( 200.0 ),
      pt1 = cms.double( 0.7 ),
      enabled = cms.bool( True ),
      pt2 = cms.double( 2.0 )
    ),
    CAPhiCut = cms.double( 0.2 ),
    useBendingCorrection = cms.bool( True ),
    fitFastCircleChi2Cut = cms.bool( True )#,
)


hltPhase2PixelTracksSeedLayers = cms.EDProducer( "SeedingLayersEDProducer",
    layerList = cms.vstring( 
        'BPix1+BPix2+BPix3+BPix4',
        'BPix1+BPix2+BPix3+FPix1_pos',
        'BPix1+BPix2+BPix3+FPix1_neg',
        'BPix1+BPix2+FPix1_pos+FPix2_pos',
        'BPix1+BPix2+FPix1_neg+FPix2_neg',
        'BPix1+FPix1_pos+FPix2_pos+FPix3_pos',
        'BPix1+FPix1_neg+FPix2_neg+FPix3_neg',
        'FPix1_pos+FPix2_pos+FPix3_pos+FPix4_pos',
        'FPix1_neg+FPix2_neg+FPix3_neg+FPix4_neg',
        'FPix2_pos+FPix3_pos+FPix4_pos+FPix5_pos',
        'FPix2_neg+FPix3_neg+FPix4_neg+FPix5_neg',
        'FPix3_pos+FPix4_pos+FPix5_pos+FPix6_pos',
        'FPix3_neg+FPix4_neg+FPix5_neg+FPix6_neg',
        'FPix4_pos+FPix5_pos+FPix6_pos+FPix7_pos',
        'FPix4_neg+FPix5_neg+FPix6_neg+FPix7_neg',
        'FPix5_pos+FPix6_pos+FPix7_pos+FPix8_pos',
        'FPix5_neg+FPix6_neg+FPix7_neg+FPix8_neg'
#      'BPix1+BPix2+BPix3+BPix4',
#      'BPix1+BPix2+BPix3+FPix1_pos',
#      'BPix1+BPix2+BPix3+FPix1_neg',
#      'BPix1+BPix2+FPix1_pos+FPix2_pos',
#      'BPix1+BPix2+FPix1_neg+FPix2_neg',
#      'BPix1+FPix1_pos+FPix2_pos+FPix3_pos',
#      'BPix1+FPix1_neg+FPix2_neg+FPix3_neg' 
    ),
    MTOB = cms.PSet(  ),
    TEC = cms.PSet(  ),
    MTID = cms.PSet(  ),
    FPix = cms.PSet( 
        HitProducer = cms.string('siPixelRecHits'), #PreSplitting'),
        TTRHBuilder = cms.string('WithTrackAngle')
#      hitErrorRPhi = cms.double( 0.0051 ),
#      TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
#      useErrorsFromParam = cms.bool( True ),
#      hitErrorRZ = cms.double( 0.0036 ),
#      HitProducer = cms.string( "hltSiPixelRecHits" )
    ),
    MTEC = cms.PSet(  ),
    MTIB = cms.PSet(  ),
    TID = cms.PSet(  ),
    TOB = cms.PSet(  ),
    BPix = cms.PSet( 
        HitProducer = cms.string('siPixelRecHits'), #PreSplitting'),
        TTRHBuilder = cms.string('WithTrackAngle')
#      hitErrorRPhi = cms.double( 0.0027 ),
#      TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
#      useErrorsFromParam = cms.bool( True ),
#      hitErrorRZ = cms.double( 0.006 ),
#      HitProducer = cms.string( "hltSiPixelRecHits" )
    ),
    TIB = cms.PSet(  )
)


hltPhase2PixelTracksTrackingRegions = cms.EDProducer( "GlobalTrackingRegionFromBeamSpotEDProducer",
    RegionPSet = cms.PSet( 
      nSigmaZ = cms.double( 4.0 ),
      beamSpot = cms.InputTag( "offlineBeamSpot" ),
      ptMin = cms.double( 0.9 ), # previous 0.8
      originRadius = cms.double( 0.02 ),
      precise = cms.bool( True )
    )
)

hltPhase2PixelTracksHitDoublets = cms.EDProducer( "HitPairEDProducer",
    trackingRegions = cms.InputTag( "hltPhase2PixelTracksTrackingRegions" ),
    layerPairs = cms.vuint32( 0, 1, 2 ),
    clusterCheck = cms.InputTag( "" ),
    produceSeedingHitSets = cms.bool( False ),
    produceIntermediateHitDoublets = cms.bool( True ),
    trackingRegionsSeedingLayers = cms.InputTag( "" ),
    maxElementTotal = cms.uint32( 50000000 ),
    maxElement = cms.uint32(50000000), # 0 ),
    seedingLayers = cms.InputTag( "hltPhase2PixelTracksSeedLayers" ) 
)

hltPhase2PixelTracks = cms.EDProducer("PixelTrackProducer",
    Cleaner = cms.string('hltPhase2PixelTrackCleanerBySharedHits'),
    passLabel = cms.string('hltPhase2PixelTracks'),
    Filter = cms.InputTag("hltPhase2PixelTrackFilterByKinematics"),
    Fitter = cms.InputTag("hltPhase2PixelFitterByHelixProjections"),
    SeedingHitSets = cms.InputTag("hltPhase2PixelTracksHitQuadruplets"),
    mightGet = cms.untracked.vstring("")#'RegionsSeedingHitSets_pixelTracksHitQuadruplets__RECO')
)

hltPhase2PixelVertices = cms.EDProducer( "PixelVertexProducer",
    WtAverage = cms.bool( True ),
    Method2 = cms.bool( True ),
    beamSpot = cms.InputTag( "offlineBeamSpot" ),
    PVcomparer = cms.PSet(  refToPSet_ = cms.string( "hltPhase2PSetPvClusterComparerForIT" ) ),
    Verbosity = cms.int32( 0 ),
    UseError = cms.bool( True ),
    TrackCollection = cms.InputTag( "hltPhase2PixelTracks" ),
    PtMin = cms.double( 1.0 ),
    NTrkMin = cms.int32( 2 ),
    ZOffset = cms.double( 5.0 ),
    Finder = cms.string( "DivisiveVertexFinder" ),
    ZSeparation = cms.double( 0.05 )
)

hltPhase2TrimmedPixelVertices = cms.EDProducer( "PixelVertexCollectionTrimmer",
    src = cms.InputTag( "hltPhase2PixelVertices" ),
    fractionSumPt2 = cms.double( 0.3 ),
    minSumPt2 = cms.double( 0.0 ),
    PVcomparer = cms.PSet(  refToPSet_ = cms.string( "hltPhase2PSetPvClusterComparerForIT" ) ),
    maxVtx = cms.uint32( 100 ) # > 200 # previous 100
)

hltPhase2PixelTracksSequence = cms.Sequence(
    hltPhase2PixelTrackFilterByKinematics + 
    hltPhase2PixelFitterByHelixProjections +  
    hltPhase2PixelTracksTrackingRegions +  # = hlt
    hltPhase2PixelTracksSeedLayers +  
    hltPhase2PixelTracksHitDoublets + 
    hltPhase2PixelTracksHitQuadruplets + 
    hltPhase2PixelTracks
)

hltPhase2PixelVerticesSequence = cms.Sequence( # pixelVertices
    hltPhase2PixelVertices + 
    hltPhase2TrimmedPixelVertices 
)
