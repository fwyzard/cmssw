import FWCore.ParameterSet.Config as cms

ecalLocalRecoRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_ecalPreshowerRecHit_*_*',
        'keep *_ecalRecHit_*_*',
        'keep *_ecalCompactTrigPrim_*_*',
        'keep *_ecalTPSkim_*_*',
        'keep EBSrFlagsSorted_ecalDigis__*',
        'keep EESrFlagsSorted_ecalDigis__*',
        'keep *_mix_EBTimeDigi_*',
        'keep *_mix_EETimeDigi_*',
        'keep *_ecalDetailedTimeRecHit_*_*'
    )
)