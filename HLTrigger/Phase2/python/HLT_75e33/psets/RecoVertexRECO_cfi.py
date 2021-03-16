import FWCore.ParameterSet.Config as cms

RecoVertexRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep  *_offlinePrimaryVertices__*',
        'keep *_offlinePrimaryVerticesWithBS_*_*',
        'keep *_offlinePrimaryVerticesFromCosmicTracks_*_*',
        'keep *_nuclearInteractionMaker_*_*',
        'keep *_generalV0Candidates_*_*',
        'keep *_inclusiveSecondaryVertices_*_*',
        'keep *_offlinePrimaryVertices4D__*',
        'keep *_offlinePrimaryVertices4DWithBS__*',
        'keep *_trackTimeValueMapProducer_*_*',
        'keep *_offlinePrimaryVertices4DnoPID__*',
        'keep *_offlinePrimaryVertices4DnoPIDWithBS__*',
        'keep *_tofPID_*_*'
    )
)