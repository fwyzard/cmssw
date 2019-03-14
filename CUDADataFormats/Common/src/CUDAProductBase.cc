#include "CUDADataFormats/Common/interface/CUDAProductBase.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

CUDAProductBase::CUDAProductBase(int device, std::shared_ptr<CUstream_st> stream):
  stream_(stream),
  device_(device)
{
  edm::Service<CUDAService> cs;
  event_ = cs->getCUDAEvent();

  // Record CUDA event to the CUDA stream. The event will become
  // "occurred" after all work queued to the stream before this
  // point has been finished.
  event_->record(stream_.get());
}
