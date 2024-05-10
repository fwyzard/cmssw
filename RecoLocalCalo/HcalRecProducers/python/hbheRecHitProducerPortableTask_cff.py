import FWCore.ParameterSet.Config as cms

# Run 3 HCAL workflow on GPU

# EventSetup modules used by HBHERecHitProducerPortable

from RecoLocalCalo.HcalRecProducers.hcalMahiConditionsESProducer_cfi import hcalMahiConditionsESProducer
from RecoLocalCalo.HcalRecProducers.hcalMahiPulseOffsetsESProducer_cfi import hcalMahiPulseOffsetsESProducer
from RecoLocalCalo.HcalRecProducers.hcalSiPMCharacteristicsESProducer_cfi import hcalSiPMCharacteristicsESProducer 
from RecoLocalCalo.HcalRecAlgos.hcalRecoParamWithPulseShapeESProducer_cfi import hcalRecoParamWithPulseShapeESProducer 

hcalMahiPulseOffSetAlpakaESRcdSource = cms.ESSource('EmptyESSource',
    recordName = cms.string('JobConfigurationGPURecord'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

# convert the HBHE digis into SoA format
from EventFilter.HcalRawToDigi.hcalDigisProducerPortable_cfi import hcalDigisProducerPortable as _hcalDigisProducerPortable
hcalDigisPortable = _hcalDigisProducerPortable.clone(
    digisLabelF01HE = "f01HEDigisGPU",
    digisLabelF5HB  = "f5HBDigisGPU",
    digisLabelF3HB  = "f3HBDigisGPU"
)

from HeterogeneousCore.AlpakaCore.functions import *
hcalDigisSerial = makeSerialClone(hcalDigisPortable)

# run the HCAL local reconstruction (MAHI) on GPU
from RecoLocalCalo.HcalRecProducers.hbheRecHitProducerPortable_cfi import hbheRecHitProducerPortable as _hbheRecHitProducerPortable
hbheRecHitProducerPortable = _hbheRecHitProducerPortable.clone(
    digisLabelF01HE = ("hcalDigisPortable", "f01HEDigisGPU"),
    digisLabelF5HB = ("hcalDigisPortable", "f5HBDigisGPU"),
    digisLabelF3HB = ("hcalDigisPortable","f3HBDigisGPU"),
    recHitsLabelM0HBHE = "",
    mahiPulseOffSets = "hcalMahiPulseOffsetsESProducer:"
)
hbheRecHitProducerSerial = makeSerialClone(hbheRecHitProducerPortable,
    digisLabelF01HE = ("hcalDigisSerial","f01HEDigisGPU"),
    digisLabelF5HB = ("hcalDigisSerial","f5HBDigisGPU"),
    digisLabelF3HB = ("hcalDigisSerial","f3HBDigisGPU")
)

# Tasks and Sequences
hbheRecHitProducerPortableTask = cms.Task(
    hcalMahiConditionsESProducer,
    hcalMahiPulseOffSetAlpakaESRcdSource,
    hcalMahiPulseOffsetsESProducer,
    hcalRecoParamWithPulseShapeESProducer,
    hcalSiPMCharacteristicsESProducer,
    hcalDigisPortable,
    hcalDigisSerial,
    hbheRecHitProducerPortable,
    hbheRecHitProducerSerial
)

hbheRecHitProducerPortableSequence = cms.Sequence(hbheRecHitProducerPortableTask)
