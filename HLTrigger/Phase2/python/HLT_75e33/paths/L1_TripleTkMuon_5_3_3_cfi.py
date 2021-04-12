import FWCore.ParameterSet.Config as cms

from ..modules.hltBoolEnd_cfi import *
from ..modules.hltL1SingleMuFiltered5_cfi import *
from ..modules.hltL1TripleMuFiltered3_cfi import *
from ..modules.hltTriggerType_cfi import *
from ..modules.hltTripleMuon3DR0_cfi import *
from ..modules.hltTripleMuon3DZ1p0_cfi import *
from ..modules.hltScalersRawToDigi_cfi import *
from ..modules.offlineBeamSpot_cfi import *

L1_TripleTkMuon_5_3_3 = cms.Path(hltTriggerType+hltL1TripleMuFiltered3+hltL1SingleMuFiltered5+hltTripleMuon3DZ1p0+hltTripleMuon3DR0+hltBoolEnd, cms.Task(hltScalersRawToDigi, offlineBeamSpot))
