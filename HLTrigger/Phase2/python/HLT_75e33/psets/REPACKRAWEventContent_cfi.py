import FWCore.ParameterSet.Config as cms

REPACKRAWEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'drop *',
        'drop FEDRawDataCollection_*_*_*',
        'keep FEDRawDataCollection_rawDataRepacker_*_*',
        'keep FEDRawDataCollection_virginRawDataRepacker_*_*',
        'keep FEDRawDataCollection_rawDataReducedFormat_*_*',
        'keep  FEDRawDataCollection_rawDataCollector_*_*',
        'keep  FEDRawDataCollection_source_*_*',
        'drop *_hlt*_*_*',
        'keep FEDRawDataCollection_rawDataCollector_*_*',
        'keep FEDRawDataCollection_source_*_*',
        'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
        'keep edmTriggerResults_*_*_*',
        'keep triggerTriggerEvent_*_*_*',
        'keep *_hltFEDSelectorL1_*_*',
        'keep *_hltScoutingCaloPacker_*_*',
        'keep *_hltScoutingEgammaPacker_*_*',
        'keep *_hltScoutingMuonPackerCalo_*_*',
        'keep *_hltScoutingMuonPacker_*_*',
        'keep *_hltScoutingPFPacker_*_*',
        'keep *_hltScoutingPrimaryVertexPackerCaloMuon_*_*',
        'keep *_hltScoutingPrimaryVertexPacker_*_*',
        'keep *_hltScoutingTrackPacker_*_*',
        'keep edmTriggerResults_*_*_*',
        'drop FEDRawDataCollection_source_*_*',
        'drop FEDRawDataCollection_rawDataCollector_*_*'
    ),
    splitLevel = cms.untracked.int32(0)
)