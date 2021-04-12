import FWCore.ParameterSet.Config as cms

from ..modules.hltBoolEnd_cfi import *
from ..modules.hltL1TkSingleMuFiltered22_cfi import *
from ..modules.hltTriggerType_cfi import *
from ..modules.hltScalersRawToDigi_cfi import *
from ..modules.offlineBeamSpot_cfi import *

L1_SingleTkMuon_22 = cms.Path(hltTriggerType+hltL1TkSingleMuFiltered22+hltBoolEnd, cms.Task(hltScalersRawToDigi, offlineBeamSpot))
