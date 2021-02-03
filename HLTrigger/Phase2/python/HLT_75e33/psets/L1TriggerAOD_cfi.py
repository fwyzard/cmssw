import FWCore.ParameterSet.Config as cms

L1TriggerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*', 
        'keep *_l1GtRecord_*_*', 
        'keep *_l1GtTriggerMenuLite_*_*', 
        'keep *_conditionsInEdm_*_*', 
        'keep *_l1extraParticles_*_*', 
        'keep *_l1L1GtObjectMap_*_*', 
        'keep LumiSummary_lumiProducer_*_*', 
        'keep *_gtStage2Digis_*_*', 
        'keep *_gmtStage2Digis_*_*', 
        'keep *_caloStage2Digis_*_*'
    )
)