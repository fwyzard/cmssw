import FWCore.ParameterSet.Config as cms

OnlineMetaDataContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep CTPPSRecord_onlineMetaDataDigis_*_*', 
        'keep DCSRecord_onlineMetaDataDigis_*_*', 
        'keep OnlineLuminosityRecord_onlineMetaDataDigis_*_*', 
        'keep recoBeamSpot_onlineMetaDataDigis_*_*'
    )
)