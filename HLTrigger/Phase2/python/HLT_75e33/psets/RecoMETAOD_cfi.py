import FWCore.ParameterSet.Config as cms

RecoMETAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
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