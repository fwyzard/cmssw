import FWCore.ParameterSet.Config as cms

EITopPAGEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_pfIsolatedElectronsEI_*_*', 
        'keep *_pfIsolatedMuonsEI_*_*', 
        'keep recoPFJets_pfJetsEI_*_*', 
        'keep *_pfCombinedInclusiveSecondaryVertexV2BJetTagsEI_*_*', 
        'keep recoPFTaus_pfTausEI_*_*', 
        'keep recoPFTauDiscriminator_pfTausDiscriminationByDecayModeFinding_*_*', 
        'keep recoSingleTauDiscriminatorContaineredmValueMap_pfTausDiscriminationByIsolation_*_*', 
        'keep recoPFMETs_pfMetEI_*_*'
    )
)