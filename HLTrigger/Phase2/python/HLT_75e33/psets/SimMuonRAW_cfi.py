import FWCore.ParameterSet.Config as cms

SimMuonRAW = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep StripDigiSimLinkedmDetSetVector_simMuonCSCDigis_*_*', 
        'keep CSCDetIdCSCComparatorDigiMuonDigiCollection_simMuonCSCDigis_*_*', 
        'keep DTLayerIdDTDigiSimLinkMuonDigiCollection_simMuonDTDigis_*_*', 
        'keep RPCDigiSimLinkedmDetSetVector_simMuonRPCDigis_*_*', 
        'keep *_simMuonCSCDigis_*_*', 
        'keep *_simMuonRPCDigis_*_*', 
        'keep StripDigiSimLinkedmDetSetVector_simMuonGEMDigis_*_*', 
        'keep StripDigiSimLinkedmDetSetVector_simMuonME0Digis_*_*'
    )
)