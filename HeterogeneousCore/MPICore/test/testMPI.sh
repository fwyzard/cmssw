#! /bin/bash
# Shell script for testing CMSSW over MPI

mkdir -p $CMSSW_BASE/tmp/$SCRAM_ARCH/test
DIR=$(mktemp -d -p $CMSSW_BASE/tmp/$SCRAM_ARCH/test)
echo "Running MPI tests at $DIR/"
pushd $DIR > /dev/null

# start an MPI server to let independent CMSSW processes find each other
echo "Starting the Open RTE data server"
ompi-server -r server.uri -d >& ompi-server.log &
SERVER_PID=$!
disown
# wait until the ORTE server logs 'up and running'
while ! grep -q 'up and running' ompi-server.log; do
  sleep 1s
done

# Note: "mpirun --mca pmix_server_uri file:server.uri" is required to make the
# tests work inside a singularity/apptainer container. Without a container the
# cmsRun commands can be used directly.

# start the "follower" CMSSW job(s)
{
  mpirun --mca pmix_server_uri file:server.uri -n 1 -- cmsRun $CMSSW_BASE/src/HeterogeneousCore/MPICore/test/testMPISource.py >& mpisource.log
  echo $? > mpisource.status
} &

# wait to make sure the MPISource has established the connection to the ORTE server
sleep 3s

# start the "driver" CMSSW job(s)
{
  mpirun --mca pmix_server_uri file:server.uri -n 1 -- cmsRun $CMSSW_BASE/src/HeterogeneousCore/MPICore/test/testMPIDriver.py >& mpidriver.log
  echo $? > mpidriver.status
} &

# wait for all CMSSW jobs to finish
wait

# print the jobs' output and check the jobs' exit status
# pr -m -t -w 240 mpidriver.log mpisource.log
echo '============ testMPIDriver ============='
cat mpidriver.log
MPIDRIVER_STATUS=$(< mpidriver.status)
echo '========================================'
echo
echo '============ testMPISource ============='
cat mpisource.log
MPISOURCE_STATUS=$(< mpisource.status)
echo '========================================'

# stop the MPI server and cleanup the URI file
kill $SERVER_PID

popd > /dev/null
exit $((MPISOURCE_STATUS > MPIDRIVER_STATUS ? MPISOURCE_STATUS : MPIDRIVER_STATUS))
