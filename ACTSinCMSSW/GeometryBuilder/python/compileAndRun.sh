#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Running: $0 <nome_file_python>"
    exit 1
fi

PYFILE=$1

cd /afs/cern.ch/user/l/ldamenti/CMSSW_16_0_0_pre1/src
scram b -j16

cd /afs/cern.ch/user/l/ldamenti/CMSSW_16_0_0_pre1/src/ACTSinCMSSW/GeometryBuilder/python
cmsRun "$PYFILE"