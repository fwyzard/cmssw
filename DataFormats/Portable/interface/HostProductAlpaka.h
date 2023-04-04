#ifndef DataFormats_Common_interface_HostProduct_H
#define DataFormats_Common_interface_HostProduct_H

#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

// a heterogeneous unique pointer...
template <typename T>
class HostProductAlpaka {
public:
  HostProductAlpaka() = default;  // make root happy
  ~HostProductAlpaka() = default;
  HostProductAlpaka(HostProductAlpaka&&) = default;
  HostProductAlpaka& operator=(HostProductAlpaka&&) = default;

  explicit HostProductAlpaka(cms::alpakatools::host_buffer<T>&& p) : hm_ptr(std::move(p)) {}
  explicit HostProductAlpaka(std::unique_ptr<T>&& p) : std_ptr(std::move(p)) {}

  auto const* get() const { return std_ptr ? std_ptr.get() : hm_ptr.data(); }

  auto const& operator*() const { return *get(); }

  auto const* operator->() const { return get(); }

private:
  cms::alpakatools::host_buffer<T> hm_ptr;  //!
  std::unique_ptr<T> std_ptr;               //!
};

#endif
