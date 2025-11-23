#ifndef DataFormats_Common_interface_DeviceProduct_h
#define DataFormats_Common_interface_DeviceProduct_h

#include <cassert>
#include <memory>
#include <type_traits>
#include <utility>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "HeterogeneousCore/SerialisationCore/interface/MemoryCopyTraits.h"

namespace edm {
  class DeviceProductBase {
  public:
    DeviceProductBase() = default;
    ~DeviceProductBase() = default;

    // TODO: in principle this function is an implementation detail
    template <typename M>
    M const& metadata() const {
      // TODO: I believe the assertion could be removed safely after
      // the data dependence and scheduling systems would guarantee
      // that the an EDModule in a given execution space can access
      // only to the EDProducts in a memory space compatible with the
      // execution space.
      //
      // On the other hand, with Alpaka (likely with others) the
      // getSynchronized() does additional checks so the added cost is
      // probably not that much?
      assert(typeid(M) == *metadataType_);
      return *static_cast<M const*>(metadata_.get());
    }

  protected:
    template <typename M>
    explicit DeviceProductBase(std::shared_ptr<M> metadata)
        : metadata_(std::move(metadata)), metadataType_(&typeid(M)) {}

  private:
    std::shared_ptr<void const> metadata_;
    std::type_info const* metadataType_;
  };

  /**
   * A wrapper for Event Data product in device memory accompanied
   * with some device-specific metadata. Not intended to be used directly by
   * developers (except in ROOT dictionary declarations in
   * classes_def.xml similar to edm::Wrapper).
   */
  template <typename T>
  class DeviceProduct : public DeviceProductBase {
  public:
    DeviceProduct()
      requires(requires { T(); })
    = default;

    explicit DeviceProduct(edm::Uninitialized)
      requires(requires { T(edm::kUninitialized); })
        : data_{edm::kUninitialized} {}

    template <typename M, typename... Args>
    explicit DeviceProduct(std::shared_ptr<M> metadata, Args&&... args)
        : DeviceProductBase(std::move(metadata)), data_(std::forward<Args>(args)...) {}

    DeviceProduct(const DeviceProduct&) = delete;
    DeviceProduct& operator=(const DeviceProduct&) = delete;
    DeviceProduct(DeviceProduct&&) = default;
    DeviceProduct& operator=(DeviceProduct&&) = default;

    /**
     * Get the actual data product after the metadata object has
     * synchronized the access. The synchronization details depend on
     * the metadata type, which the caller must know. All the
     * arguments are passed to M::synchronize() function.
     */
    template <typename M, typename... Args>
    T const& getSynchronized(Args&&... args) const {
      auto const& md = metadata<M>();
      md.synchronize(std::forward<Args>(args)...);
      return data_;
    }
    T const& product() const { return data_; }
    T& product() { return data_; }

  private:
    T data_;  //!
  };
}  // namespace edm

namespace ngt {

  template <typename T>
    requires HasMemoryCopyTraits<T> && HasTrivialCopyProperties<T>
  struct MemoryCopyTraits<edm::DeviceProduct<T>> {
    using Properties = TrivialCopyProperties<T>;

    static Properties properties(edm::DeviceProduct<T> const& wrapper) {
      return MemoryCopyTraits<T>::properties(wrapper.product());
    }

    template <typename... Args>
    static void initialize(edm::DeviceProduct<T>& wrapper, Args&&... args) {
      MemoryCopyTraits<T>::initialize(wrapper.product(), std::forward<Args>(args)...);
    }

    static std::vector<std::span<std::byte>> regions(edm::DeviceProduct<T>& wrapper) {
      return MemoryCopyTraits<T>::regions(wrapper.product());
    }

    static std::vector<std::span<const std::byte>> regions(edm::DeviceProduct<T> const& wrapper) {
      return MemoryCopyTraits<T>::regions(wrapper.product());
    }
  };

  template <typename T>
    requires HasMemoryCopyTraits<T> && (!HasTrivialCopyProperties<T>)
  struct MemoryCopyTraits<edm::DeviceProduct<T>> {
    using Properties = void;

    template <typename... Args>
    static void initialize(edm::DeviceProduct<T>& wrapper, Args&&... args) {
      MemoryCopyTraits<T>::initialize(wrapper.product(), std::forward<Args>(args)...);
    }

    template <typename... Args>
    static std::vector<std::span<std::byte>> regions(edm::DeviceProduct<T>& wrapper, Args&&... args) {
      return MemoryCopyTraits<T>::regions(wrapper.product(), std::forward<Args>(args)...);
    }

    template <typename... Args>
    static std::vector<std::span<const std::byte>> regions(edm::DeviceProduct<T> const& wrapper, Args&&... args) {
      return MemoryCopyTraits<T>::regions(wrapper.product(), std::forward<Args>(args)...);
    }
  };

}  // namespace ngt
#endif
