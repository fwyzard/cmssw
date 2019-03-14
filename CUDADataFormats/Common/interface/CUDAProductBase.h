#ifndef CUDADataFormats_Common_CUDAProductBase_h
#define CUDADataFormats_Common_CUDAProductBase_h

#include <memory>

#include <cuda/api_wrappers.h>

/**
 * Base class for all instantiations of CUDA<T> to hold the
 * non-T-dependent members.
 */
class CUDAProductBase {
public:
  CUDAProductBase() = default; // Needed only for ROOT dictionary generation

  bool isValid() const { return stream_.get() != nullptr; }

  int device() const { return device_; }

  cuda::stream_t<> stream() const { return cuda::stream_t<>(device_, stream_.get()); }
  const std::shared_ptr<CUstream_st>& streamPtr() const { return stream_; }

  const cuda::event_t& event() const { return *event_; }
  cuda::event_t& event() { return *event_; }

protected:
  explicit CUDAProductBase(int device, std::shared_ptr<CUstream_st> stream);

private:
  // The cuda::stream_t is really shared among edm::Event products, so
  // using shared_ptr also here
  std::shared_ptr<CUstream_st> stream_; //!
  // shared_ptr because of caching in CUDAService
  std::shared_ptr<cuda::event_t> event_; //!

  int device_ = -1; //!
};

#endif
