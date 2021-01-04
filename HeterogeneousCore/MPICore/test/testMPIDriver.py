import FWCore.ParameterSet.Config as cms

process = cms.Process("MPIServer")

#process.load("FWCore.Services.Tracer_cfi")
process.load("HeterogeneousCore.MPIServices.MPIService_cfi")
process.MPIService.pmix_server_uri = 'file:server.uri'

process.source = cms.Source( "PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_11_2_0/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW-HLTDEBUG/PUpmx_112X_mcRun3_2021_realistic_v13_forPMX-v1/10000/4a5fcbd6-1716-4733-a61e-0faaf5892eb9.root'
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(1),
)

from HeterogeneousCore.MPICore.mpiDriver_cfi import mpiDriver as mpiDriver_
process.mpiDriver = mpiDriver_.clone(
  eventProducts = [ "hltGtStage2Digis" ]
)
process.path = cms.Path(process.mpiDriver)
