import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts
from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants

hltQualityCuts = PFTauQualityCuts.clone()
hltQualityCuts.signalQualityCuts.minTrackPt = cms.double(0.9)
hltQualityCuts.isolationQualityCuts.minTrackPt = cms.double(0.9)
hltQualityCuts.isolationQualityCuts.maxDeltaZ = cms.double(0.15)
hltQualityCuts.isolationQualityCuts.maxDeltaZToLeadTrack = cms.double(-1.)
hltQualityCuts.isolationQualityCuts.minTrackHits = cms.uint32(8)
hltQualityCuts.primaryVertexSrc = cms.InputTag('') # CV: to be set by user
#------------------------------------------------------------------------------------------------
# CV: fix for Phase-2 HLT tau trigger studies
#    (pT of PFCandidates within HGCal acceptance is significantly higher than track pT !!)
hltQualityCuts.leadingTrkOrPFCandOption = cms.string('minLeadTrackOrPFCand')
#------------------------------------------------------------------------------------------------

hltPFTauChargedIsoPtSum = cms.EDProducer("PFRecoTauDiscriminationByIsolation",
  PFTauProducer = cms.InputTag(''),                # CV: to be set by user
  particleFlowSrc = cms.InputTag(''),              # CV: to be set by user
  vertexSrc = cms.InputTag(''),                    # CV: to be set by user
  qualityCuts = hltQualityCuts,
  Prediscriminants = noPrediscriminants,
  ApplyDiscriminationByTrackerIsolation = cms.bool(True),
  ApplyDiscriminationByECALIsolation = cms.bool(False),
  ApplyDiscriminationByWeightedECALIsolation = cms.bool(False),
  enableHGCalWorkaround = cms.bool(True),
  WeightECALIsolation = cms.double(0.),
  minTauPtForNoIso = cms.double(-99.),
  applyOccupancyCut = cms.bool(False),
  maximumOccupancy = cms.uint32(0),
  applySumPtCut = cms.bool(False),
  maximumSumPtCut = cms.double(-1.),
  applyRelativeSumPtCut = cms.bool(False),
  relativeSumPtCut = cms.double(-1.),
  relativeSumPtOffset = cms.double(0.),
  storeRawOccupancy = cms.bool(False),
  storeRawSumPt = cms.bool(True),
  storeRawPUsumPt = cms.bool(False),
  storeRawFootprintCorrection = cms.bool(False),
  storeRawPhotonSumPt_outsideSignalCone = cms.bool(False),
  customOuterCone = cms.double(-1.),
  applyPhotonPtSumOutsideSignalConeCut = cms.bool(False),
  maxAbsPhotonSumPt_outsideSignalCone = cms.double(1.e+9),
  maxRelPhotonSumPt_outsideSignalCone = cms.double(0.1),
  applyFootprintCorrection = cms.bool(False),
  footprintCorrections = cms.VPSet(),
  applyDeltaBetaCorrection = cms.bool(False),
  deltaBetaPUTrackPtCutOverride = cms.bool(False),
  deltaBetaPUTrackPtCutOverride_val = cms.double(0.5),
  isoConeSizeForDeltaBeta = cms.double(0.5),
  deltaBetaFactor = cms.string("0.38"),
  applyRhoCorrection = cms.bool(False),
  rhoProducer = cms.InputTag("NotUsed"),
  rhoConeSize = cms.double(0.357),
  rhoUEOffsetCorrection = cms.double(0.),
  UseAllPFCandsForWeights = cms.bool(False),
  verbosity = cms.int32(0)
)
