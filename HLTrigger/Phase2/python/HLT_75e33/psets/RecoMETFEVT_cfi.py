import FWCore.ParameterSet.Config as cms

RecoMETFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *HaloData_*_*_*', 
        'keep *BeamHaloSummary_BeamHaloSummary_*_*', 
        'keep recoHcalNoiseRBXs_hcalnoise_*_*', 
        'keep recoEcalHaloData_EcalHaloData_*_*', 
        'keep recoHcalHaloData_HcalHaloData_*_*', 
        'keep recoCaloMETs_caloMet_*_*', 
        'keep recoCaloMETs_caloMetBE_*_*', 
        'keep recoCaloMETs_caloMetBEFO_*_*', 
        'keep recoCaloMETs_caloMetM_*_*', 
        'keep recoPFMETs_pfMet_*_*', 
        'keep recoPFMETs_pfChMet_*_*', 
        'keep floatedmValueMap_puppiNoLep_*_*', 
        'keep recoPFMETs_pfMetPuppi_*_*', 
        'keep recoMuonMETCorrectionDataedmValueMap_muonMETValueMapProducer_*_*', 
        'keep HcalNoiseSummary_hcalnoise_*_*', 
        'keep recoGlobalHaloData_GlobalHaloData_*_*', 
        'keep recoCSCHaloData_CSCHaloData_*_*', 
        'keep recoBeamHaloSummary_BeamHaloSummary_*_*'
    )
)