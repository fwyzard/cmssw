import FWCore.ParameterSet.Config as cms

HLTScouting = cms.PSet(
    outputCommands = cms.vstring(
        'keep *_hltFEDSelectorL1_*_*',
        'keep *_hltScoutingCaloPacker_*_*',
        'keep *_hltScoutingEgammaPacker_*_*',
        'keep *_hltScoutingMuonPackerCalo_*_*',
        'keep *_hltScoutingMuonPacker_*_*',
        'keep *_hltScoutingPFPacker_*_*',
        'keep *_hltScoutingPrimaryVertexPackerCaloMuon_*_*',
        'keep *_hltScoutingPrimaryVertexPacker_*_*',
        'keep *_hltScoutingTrackPacker_*_*',
        'keep edmTriggerResults_*_*_*'
    )
)