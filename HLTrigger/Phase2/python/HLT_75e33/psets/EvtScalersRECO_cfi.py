import FWCore.ParameterSet.Config as cms

EvtScalersRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep L1AcceptBunchCrossings_scalersRawToDigi_*_*', 
        'keep L1TriggerScalerss_scalersRawToDigi_*_*', 
        'keep Level1TriggerScalerss_scalersRawToDigi_*_*', 
        'keep LumiScalerss_scalersRawToDigi_*_*', 
        'keep BeamSpotOnlines_scalersRawToDigi_*_*', 
        'keep DcsStatuss_scalersRawToDigi_*_*', 
        'keep DcsStatuss_hltScalersRawToDigi_*_*'
    )
)