import FWCore.ParameterSet.Config as cms

FASTPUEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_fastSimProducer_*_*',
        'keep *_MuonSimHits_*_*',
        'drop *_fastSimProducer_VertexTypes_*',
        'keep *_generalTracksBeforeMixing_*_*',
        'drop *_generalTracksBeforeMixing_MVAValues_*',
        'drop *_generalTracksBeforeMixing_QualityMasks_*',
        'keep edmHepMCProduct_generatorSmeared_*_*'
    )
)