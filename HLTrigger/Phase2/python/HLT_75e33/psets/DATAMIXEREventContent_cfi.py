import FWCore.ParameterSet.Config as cms

DATAMIXEREventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep CSCDetIdCSCALCTDigiMuonDigiCollection_muonCSCDigis_MuonCSCALCTDigi_*',
        'keep CSCDetIdCSCCLCTDigiMuonDigiCollection_muonCSCDigis_MuonCSCCLCTDigi_*',
        'keep CSCDetIdCSCComparatorDigiMuonDigiCollection_muonCSCDigis_MuonCSCComparatorDigi_*',
        'keep CSCDetIdCSCCorrelatedLCTDigiMuonDigiCollection_csctfDigis_*_*',
        'keep CSCDetIdCSCCorrelatedLCTDigiMuonDigiCollection_muonCSCDigis_MuonCSCCorrelatedLCTDigi_*',
        'keep CSCDetIdCSCRPCDigiMuonDigiCollection_muonCSCDigis_MuonCSCRPCDigi_*',
        'keep CSCDetIdCSCStripDigiMuonDigiCollection_muonCSCDigis_MuonCSCStripDigi_*',
        'keep CSCDetIdCSCWireDigiMuonDigiCollection_muonCSCDigis_MuonCSCWireDigi_*',
        'keep DTLayerIdDTDigiMuonDigiCollection_muonDTDigis_*_*',
        'keep PixelDigiedmDetSetVector_siPixelDigis_*_*',
        'keep SiStripDigiedmDetSetVector_siStripDigis_*_*',
        'keep RPCDetIdRPCDigiMuonDigiCollection_muonRPCDigis_*_*',
        'keep HBHEDataFramesSorted_hcalDigis_*_*',
        'keep HFDataFramesSorted_hcalDigis_*_*',
        'keep HODataFramesSorted_hcalDigis_*_*',
        'keep QIE10DataFrameHcalDataFrameContainer_hcalDigis_*_*',
        'keep QIE11DataFrameHcalDataFrameContainer_hcalDigis_*_*',
        'keep ZDCDataFramesSorted_hcalDigis_*_*',
        'keep CastorDataFramesSorted_castorDigis_*_*',
        'keep EBDigiCollection_ecalDigis_*_*',
        'keep EEDigiCollection_ecalDigis_*_*',
        'keep ESDigiCollection_ecalPreshowerDigis_*_*'
    ),
    splitLevel = cms.untracked.int32(0)
)