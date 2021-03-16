import FWCore.ParameterSet.Config as cms

RecoMTDRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_trackExtenderWithMTD_*_*',
        'keep *_mtdTrackQualityMVA_*_*'
    )
)