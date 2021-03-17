import FWCore.ParameterSet.Config as cms

SimMuonRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep StripDigiSimLinkedmDetSetVector_simMuonCSCDigis_*_*', 
        'keep DTLayerIdDTDigiSimLinkMuonDigiCollection_simMuonDTDigis_*_*', 
        'keep RPCDigiSimLinkedmDetSetVector_simMuonRPCDigis_*_*', 
        'keep *_muonSimClassifier_*_*', 
        'keep StripDigiSimLinkedmDetSetVector_simMuonGEMDigis_*_*', 
        'keep StripDigiSimLinkedmDetSetVector_simMuonME0Digis_*_*'
    )
)