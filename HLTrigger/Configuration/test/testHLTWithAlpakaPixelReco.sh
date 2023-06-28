#!/bin/bash

https_proxy=http://cmsproxy.cms:3128/ \
hltGetConfiguration /frozen/2023/2e34/v1.2/HLT \
  --globaltag 130X_dataRun3_HLT_v2 \
  --data \
  --unprescale \
  --output all \
  --max-events 100 \
  --paths DQM_PixelReco*,*DQMGPUvsCPU* \
  --input /store/data/Run2023C/EphemeralHLTPhysics0/RAW/v1/000/368/822/00000/6e1268da-f96a-49f6-a5f0-89933142dd89.root \
  --customise \
HLTrigger/Configuration/customizeHLTforPatatrack.customiseHLTforAlpakaPixelReco,\
HLTrigger/Configuration/customizeHLTforPatatrack.customiseHLTforTestingDQMGPUvsCPUPixelOnlyUpToLocal \
  > hlt.py

cat <<EOF >> hlt.py
process.hltOutputDQMGPUvsCPU.fileName = '___JOBNAME___.root'
EOF

JOBNAME=hlt0
sed "s|___JOBNAME___|${JOBNAME}|" hlt.py > "${JOBNAME}".py
edmConfigDump --prune "${JOBNAME}".py > "${JOBNAME}"_dump.py
echo "${JOBNAME}" ... && cmsRun "${JOBNAME}".py &> "${JOBNAME}".log
