
import FWCore.ParameterSet.Config as cms
#from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA


#we will need to write a simple module for testing


#config name
process = cms.Process('Writer')

process.load('HeterogeneousCore.MPIServices.MPIService_cfi')

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.MPIService = dict()


process.source = cms.Source('MPISource')

