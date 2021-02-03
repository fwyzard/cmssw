import FWCore.ParameterSet.Config as cms

RecoMTDFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_trackExtenderWithMTD_*_*', 
        'keep *_mtdTrackQualityMVA_*_*'
    )
)