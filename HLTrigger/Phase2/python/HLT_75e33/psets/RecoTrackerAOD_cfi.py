import FWCore.ParameterSet.Config as cms

RecoTrackerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep recoTracks_generalTracks_*_*', 
        'keep recoTracks_conversionStepTracks_*_*', 
        'keep recoTracks_beamhaloTracks_*_*', 
        'keep recoTracks_ctfPixelLess_*_*', 
        'keep *_dedxHarmonic2_*_*', 
        'keep *_dedxPixelHarmonic2_*_*', 
        'keep *_dedxHitInfo_*_*', 
        'keep *_trackExtrapolator_*_*', 
        'keep *_generalTracks_MVAValues_*', 
        'keep *_generalTracks_MVAVals_*'
    )
)