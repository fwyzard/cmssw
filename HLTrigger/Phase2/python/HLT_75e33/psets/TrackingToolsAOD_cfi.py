import FWCore.ParameterSet.Config as cms

TrackingToolsAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep recoTracks_GsfGlobalElectronTest_*_*', 
        'keep recoGsfTracks_electronGsfTracks_*_*', 
        'keep recoGsfTracks_electronGsfTracksFromMultiCl_*_*'
    )
)