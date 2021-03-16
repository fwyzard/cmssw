import FWCore.ParameterSet.Config as cms

RecoLocalMuonAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_dt4DSegments_*_*',
        'keep *_dt4DCosmicSegments_*_*',
        'keep *_cscSegments_*_*',
        'keep *_rpcRecHits_*_*',
        'keep *_gemRecHits_*_*',
        'keep *_gemSegments_*_*',
        'keep *_me0RecHits_*_*',
        'keep *_me0Segments_*_*'
    )
)