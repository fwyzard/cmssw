import FWCore.ParameterSet.Config as cms
#from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA


#we will need to write a simple module for testing


#config name
process = cms.Process('Writer')

process.load('HeterogeneousCore.MPIServices.MPIService_cfi')

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.MPIService = dict()


process.source = cms.Source('EmptySource',
         firstRun = cms.untracked.uint32(100),
    numberEventsInRun = cms.untracked.uint32(10))


#package or modules to load (we dont need for MPI 
#process.load('Configuration.StandardSequences.Accelerators_cff')

# enable logging for the TestPortableAnalyzer, we need it later ok with cout/cin now
#process.MessageLogger.TestPortableAnalyzer = cms.      untracked.PSet()

process.mpiDriver = cms.EDProducer("MPIDriver", service=cms.untracked.string("mpi_server"))

# run the producer on the cpu
tagID = 1
#process.driverController = cms.EDProducer('MPIDriverController', communicator=cms.InputTag("mpiDriver")); 
process.recv= cms.EDProducer('MPIRecv', communicator=cms.InputTag("mpiDriver"), tagID=cms.InputTag("mpiDriver"), userTagID=cms.untracked.int32(tagID))

tagID += 100

process.intRecv= cms.EDProducer('MPIRecvInt', communicator=cms.InputTag("mpiDriver"), tagID=cms.InputTag("mpiDriver"), userTagID=cms.untracked.int32(tagID))
tagID += 100 
process.processData = cms.EDProducer('processData', recvVectorData=cms.InputTag('recv'), recvIntData=cms.InputTag('recv'))

process.sender= cms.EDProducer('MPISend', communicator=cms.InputTag("mpiDriver"), tagID=cms.InputTag("mpiDriver"), intData=cms.InputTag('processData'), userTagID=cms.untracked.int32(tagID), vectorData=cms.InputTag("processData"))

tagID+= 100

process.intSender= cms.EDProducer('MPISendInt', communicator=cms.InputTag("mpiDriver"), tagID=cms.InputTag("mpiDriver"), intData=cms.InputTag('processData'), userTagID=cms.untracked.int32(tagID))


#process.send= cms.EDProducer('MPISend', communicator=cms.InputTag("mpiSource"))


process.producer_task = cms.Task(process.mpiDriver)
process.process_path = cms.Path( cms.wait(process.recv+process.intRecv) *process.processData* cms.wait(process.sender+process.intSender) , process.producer_task) 
#process.process_path = cms.Path( cms.wait(process.recv) *process.processData* cms.wait(process.sender) , process.producer_task) 


process.maxEvents.input =100

process.options.numberOfStreams = 10
process.options.numberOfThreads = 10
process.options.numberOfConcurrentRuns = 1

