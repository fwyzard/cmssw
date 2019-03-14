#ifndef HeterogeneousCore_CUDACore_CUDAContextToken_h
#define HeterogeneousCore_CUDACore_CUDAContextToken_h

#include <memory>

#include <cuda.h>

/**
 * The purpose of this class is to deliver the device and CUDA stream
 * information from ExternalWork's acquire() to producer() via a
 * member/StreamCache variable.
 */
class CUDAContextToken {
public:
  CUDAContextToken() = default;
  ~CUDAContextToken() = default;

  CUDAContextToken(const CUDAContextToken&) = delete;
  CUDAContextToken& operator=(const CUDAContextToken&) = delete;
  CUDAContextToken(CUDAContextToken&&) = default;
  CUDAContextToken& operator=(CUDAContextToken&& other) = default;

private:
  friend class CUDAScopedContext;

  explicit CUDAContextToken(int device, std::shared_ptr<CUstream_st> stream):
    stream_(std::move(stream)),
    device_(device)
  {}

  int device() { return device_; }

  std::shared_ptr<CUstream_st>&& streamPtr() {
    return std::move(stream_);
  }

  std::shared_ptr<CUstream_st> stream_;
  int device_;
};

#endif
