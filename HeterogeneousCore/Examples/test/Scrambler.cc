#include <string>
#include <cuda_runtime.h>

#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "ScramblerKernel.cuh"

class Scrambler: public edm::stream::EDAnalyzer<> {
public:
  Scrambler(edm::ParameterSet const& config) :
    message_(config.getUntrackedParameter<std::string>("message"))
  {
    cudaCheck(cudaMalloc(& buffer_, message_.size()));
    cudaCheck(cudaMemcpy(buffer_, message_.data(), message_.size(), cudaMemcpyDefault));
  }

  ~Scrambler()
  {
    cudaCheck(cudaFree(buffer_));
  }

  void analyze(edm::Event const&, edm::EventSetup const&)
  {
    scrambler_wrapper(buffer_, message_.size());
  }

private:
  std::string message_;
  char *      buffer_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(Scrambler);
