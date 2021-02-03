import FWCore.ParameterSet.Config as cms

RecoMuonIsolationParamGlobal = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_muParamGlobalIsoDepositGsTk_*_*', 
        'keep *_muParamGlobalIsoDepositCalEcal_*_*', 
        'keep *_muParamGlobalIsoDepositCalHcal_*_*', 
        'keep *_muParamGlobalIsoDepositCtfTk_*_*', 
        'keep *_muParamGlobalIsoDepositCalByAssociatorTowers_*_*', 
        'keep *_muParamGlobalIsoDepositCalByAssociatorHits_*_*', 
        'keep *_muParamGlobalIsoDepositJets_*_*'
    )
)