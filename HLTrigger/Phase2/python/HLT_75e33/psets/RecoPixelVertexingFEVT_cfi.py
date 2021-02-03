import FWCore.ParameterSet.Config as cms

RecoPixelVertexingFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_pixelTracks_*_*', 
        'keep *_pixelVertices_*_*'
    )
)