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
    numberEventsInRun = cms.untracked.uint32(2))


#package or modules to load (we dont need for MPI 
#process.load('Configuration.StandardSequences.Accelerators_cff')

# enable logging for the TestPortableAnalyzer, we need it later ok with cout/cin now
#process.MessageLogger.TestPortableAnalyzer = cms.      untracked.PSet()

process.controller = cms.EDProducer("MPIController", service=cms.untracked.string("mpi_server"))

# run the producer on the cpu
process.generator = cms.EDProducer('GenerateDummyData') 
process.sender= cms.EDProducer('MPISend', communicator=cms.InputTag("controller"), incomingData=cms.InputTag("generator"))
process.recv = cms.EDProducer('MPIRecv', controller=cms.InputTag("sender")) 
process.compare = cms.EDProducer('CompareData', sourceData=cms.InputTag("recv"), controllerData=cms.InputTag("generator"))


process.producer_task = cms.Task(process.controller)

#process.process_path = cms.Path(process.controller)
process.process_path = cms.Path(cms.wait(process.generator+ process.sender + process.recv)+process.compare, process.producer_task)

#process.process_path = cms.Path(process.generator+ process.sender , process.producer_task)

#process.serial_path = cms.Path(   process.testProducerSerial +  process.testAnalyzerSerial)

#process.output_path = cms.EndPath(process.output)

process.maxEvents.input =30 

process.options.numberOfStreams = 1 
process.options.numberOfThreads = 1 
process.options.numberOfConcurrentRuns = 1
