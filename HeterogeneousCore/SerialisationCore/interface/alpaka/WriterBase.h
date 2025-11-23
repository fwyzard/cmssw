#ifndef TrivialSerialisation_Common_interface_alpaka_WriterBase_h
#define TrivialSerialisation_Common_interface_alpaka_WriterBase_h

#include <cstddef>
#include <span>
#include <vector>

#include "DataFormats/Common/interface/WrapperBase.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/SerialisationCore/interface/AnyBuffer.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt {

  class WriterBase {
  public:
    WriterBase() = default;
    virtual ~WriterBase() = default;

    virtual void initialize(Queue& queue, ::ngt::AnyBuffer const& args) = 0;
    virtual ::ngt::AnyBuffer uninitialized_parameters() const = 0;
    virtual std::vector<std::span<std::byte>> regions() = 0;
    virtual void finalize() = 0;

    std::unique_ptr<edm::WrapperBase> get() { return std::move(ptr_); }

  protected:
    std::unique_ptr<edm::WrapperBase> ptr_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt

#endif  // TrivialSerialisation_Common_interface_alpaka_WriterBase_h
