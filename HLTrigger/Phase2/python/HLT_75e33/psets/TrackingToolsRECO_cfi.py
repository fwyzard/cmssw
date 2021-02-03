import FWCore.ParameterSet.Config as cms

TrackingToolsRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_CkfElectronCandidates_*_*', 
        'keep *_GsfGlobalElectronTest_*_*', 
        'keep *_electronMergedSeeds_*_*', 
        'keep recoGsfTrackExtras_electronGsfTracks_*_*', 
        'keep recoTrackExtras_electronGsfTracks_*_*', 
        'keep TrackingRecHitsOwned_electronGsfTracks_*_*', 
        'keep recoTracks_GsfGlobalElectronTest_*_*', 
        'keep recoGsfTracks_electronGsfTracks_*_*', 
        'keep recoGsfTracks_electronGsfTracksFromMultiCl_*_*', 
        'keep recoGsfTracks_electronGsfTracksFromMultiCl_*_*', 
        'keep recoGsfTrackExtras_electronGsfTracksFromMultiCl_*_*', 
        'keep recoTrackExtras_electronGsfTracksFromMultiCl_*_*', 
        'keep TrackingRecHitsOwned_electronGsfTracksFromMultiCl_*_*', 
        'keep *_electronMergedSeedsFromMultiCl_*_*'
    )
)