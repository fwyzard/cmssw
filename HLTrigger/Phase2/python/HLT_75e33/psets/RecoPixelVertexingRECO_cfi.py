import FWCore.ParameterSet.Config as cms

RecoPixelVertexingRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_pixelTracks_*_*', 
        'keep *_pixelVertices_*_*'
    )
)