import FWCore.ParameterSet.Config as cms

from ..modules.hltBoolEnd_cfi import *
from ..modules.hltDoubleMuon7DZ1p0_cfi import *
from ..modules.hltL1TkDoubleMuFiltered7_cfi import *
from ..modules.hltL1TkSingleMuFiltered15_cfi import *
from ..modules.hltTriggerType_cfi import *
from ..modules.hltScalersRawToDigi_cfi import *
from ..modules.offlineBeamSpot_cfi import *

L1_DoubleTkMuon_17_8 = cms.Path(hltTriggerType+hltL1TkDoubleMuFiltered7+hltL1TkSingleMuFiltered15+hltDoubleMuon7DZ1p0+hltBoolEnd, cms.Task(hltScalersRawToDigi, offlineBeamSpot))
