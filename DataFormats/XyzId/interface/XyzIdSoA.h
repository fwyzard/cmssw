#ifndef DataFormats_XyzId_interface_XyzIdSoA_h
#define DataFormats_XyzId_interface_XyzIdSoA_h

#ifdef DEBUG_SOA_CTOR_DTOR
#include <iostream>
#endif

#include "DataFormats/XyzId/interface/SoACommon.h"
#include "DataFormats/XyzId/interface/SoALayout.h"
#include "DataFormats/XyzId/interface/SoAView.h"

// XXX Addition: forward declaration of trivial view, new size type
// XXX translation of typedef in template for ROOT (cms::soa::byte_size_type = size_t)
template <size_t ALIGNMENT = cms::soa::CacheLineSize::defaultSize,
          bool ALIGNMENT_ENFORCEMENT = cms::soa::AlignmentEnforcement::Relaxed,
          bool RESTRICT_QUALIFY = cms::soa::RestrictQualify::Disabled,
          bool RANGE_CHECKING = cms::soa::RangeChecking::Disabled>
struct SoAViewTemplate;

// XXX new size type.
template <size_t ALIGNMENT = cms::soa::CacheLineSize::defaultSize, bool ALIGNMENT_ENFORCEMENT = cms::soa::AlignmentEnforcement::Relaxed>
struct SoALayoutTemplate {
  using self_type = SoALayoutTemplate;

  // XXX size types,
  using size_type = cms::soa::size_type;
  using byte_size_type = cms::soa::byte_size_type;
  using AlignmentEnforcement = cms::soa::AlignmentEnforcement;
  constexpr static byte_size_type defaultAlignment = 128;
  constexpr static byte_size_type alignment = ALIGNMENT;
  constexpr static bool alignmentEnforcement = ALIGNMENT_ENFORCEMENT;
  constexpr static byte_size_type conditionalAlignment = alignmentEnforcement == cms::soa::AlignmentEnforcement::Enforced ? alignment : 0;
  template <cms::soa::SoAColumnType COLUMN_TYPE, class C>
  using SoAValueWithConf = cms::soa::SoAValue<COLUMN_TYPE, C, conditionalAlignment>;
  template <cms::soa::SoAColumnType COLUMN_TYPE, class C>
  using SoAConstValueWithConf = cms::soa::SoAConstValue<COLUMN_TYPE, C, conditionalAlignment>;
  void soaToStreamInternal(std::ostream& os) const {
    os << "SoALayoutTemplate"
          "("
       << nElements_ << " elements, byte alignement= " << alignment << ", @" << mem_ << "): " << std::endl;
    os << "  sizeof("
          "SoALayoutTemplate"
          "): "
       << sizeof(SoALayoutTemplate) << std::endl;
    // XXX size types,
    byte_size_type offset = 0;
    os << " Column "
          "x"
          " at offset "
       << offset << " has size " << sizeof(double) * nElements_ << " and padding " << (((nElements_ * sizeof(double) - 1) / alignment) + 1) * alignment - (sizeof(double) * nElements_) << std::endl;
    offset += (((nElements_ * sizeof(double) - 1) / alignment) + 1) * alignment;
    os << " Column "
          "y"
          " at offset "
       << offset << " has size " << sizeof(double) * nElements_ << " and padding " << (((nElements_ * sizeof(double) - 1) / alignment) + 1) * alignment - (sizeof(double) * nElements_) << std::endl;
    offset += (((nElements_ * sizeof(double) - 1) / alignment) + 1) * alignment;
    os << " Column "
          "z"
          " at offset "
       << offset << " has size " << sizeof(double) * nElements_ << " and padding " << (((nElements_ * sizeof(double) - 1) / alignment) + 1) * alignment - (sizeof(double) * nElements_) << std::endl;
    offset += (((nElements_ * sizeof(double) - 1) / alignment) + 1) * alignment;
    os << " Column "
          "id"
          " at offset "
       << offset << " has size " << sizeof(int32_t) * nElements_ << " and padding " << (((nElements_ * sizeof(int32_t) - 1) / alignment) + 1) * alignment - (sizeof(int32_t) * nElements_) << std::endl;
    offset += (((nElements_ * sizeof(int32_t) - 1) / alignment) + 1) * alignment;
    os << "Final offset = " << offset << " computeDataSize(...): " << computeDataSize(nElements_) << std::endl;
    os << std::endl;
  }

  // Size type
  // XXX move from size_t to cms_uint32_t.
  // XXX size types,
  static byte_size_type computeDataSize(size_type nElements) {
    // XXX size types,
    byte_size_type ret = 0;
    ret += (((nElements * sizeof(double) - 1) / alignment) + 1) * alignment;
    ret += (((nElements * sizeof(double) - 1) / alignment) + 1) * alignment;
    ret += (((nElements * sizeof(double) - 1) / alignment) + 1) * alignment;
    ret += (((nElements * sizeof(int32_t) - 1) / alignment) + 1) * alignment;
    return ret;
  }

  struct SoAMetadata {
    friend SoALayoutTemplate;
    // XXX size types,
    inline size_type size() const { return parent_.nElements_; }
    // XXX size types,
    inline byte_size_type byteSize() const { return parent_.byteSize_; }
    // XXX size types,
    inline byte_size_type alignment() const { return SoALayoutTemplate::alignment; }
    inline std::byte* data() { return parent_.mem_; }
    inline const std::byte* data() const { return parent_.mem_; }
    inline std::byte* nextByte() const { return parent_.mem_ + parent_.byteSize_; }
    inline SoALayoutTemplate cloneToNewAddress(std::byte* addr) const { return SoALayoutTemplate(addr, parent_.nElements_); }
    using ParametersTypeOf_x = cms::soa::SoAParameters_ColumnType<cms::soa::SoAColumnType::column>::DataType<double>;
    inline ParametersTypeOf_x parametersOf_x() const { return ParametersTypeOf_x(parent_.x_); }
    inline double const* addressOf_x() const { return parent_.soaMetadata().parametersOf_x().addr_; }
    inline double* addressOf_x() { return parent_.soaMetadata().parametersOf_x().addr_; }
    // XXX size types,
    inline byte_size_type xPitch() const { return (((parent_.nElements_ * sizeof(double) - 1) / ParentClass::alignment) + 1) * ParentClass::alignment; }
    using TypeOf_x = double;
    constexpr static cms::soa::SoAColumnType ColumnTypeOf_x = cms::soa::SoAColumnType::column;
    using ParametersTypeOf_y = cms::soa::SoAParameters_ColumnType<cms::soa::SoAColumnType::column>::DataType<double>;
    inline ParametersTypeOf_y parametersOf_y() const { return ParametersTypeOf_y(parent_.y_); }
    inline double const* addressOf_y() const { return parent_.soaMetadata().parametersOf_y().addr_; }
    inline double* addressOf_y() { return parent_.soaMetadata().parametersOf_y().addr_; }
    // XXX size types,
    inline byte_size_type yPitch() const { return (((parent_.nElements_ * sizeof(double) - 1) / ParentClass::alignment) + 1) * ParentClass::alignment; }
    using TypeOf_y = double;
    constexpr static cms::soa::SoAColumnType ColumnTypeOf_y = cms::soa::SoAColumnType::column;
    using ParametersTypeOf_z = cms::soa::SoAParameters_ColumnType<cms::soa::SoAColumnType::column>::DataType<double>;
    inline ParametersTypeOf_z parametersOf_z() const { return ParametersTypeOf_z(parent_.z_); }
    inline double const* addressOf_z() const { return parent_.soaMetadata().parametersOf_z().addr_; }
    inline double* addressOf_z() { return parent_.soaMetadata().parametersOf_z().addr_; }
    // XXX size types,
    inline byte_size_type zPitch() const { return (((parent_.nElements_ * sizeof(double) - 1) / ParentClass::alignment) + 1) * ParentClass::alignment; }
    using TypeOf_z = double;
    constexpr static cms::soa::SoAColumnType ColumnTypeOf_z = cms::soa::SoAColumnType::column;
    using ParametersTypeOf_id = cms::soa::SoAParameters_ColumnType<cms::soa::SoAColumnType::column>::DataType<int32_t>;
    inline ParametersTypeOf_id parametersOf_id() const { return ParametersTypeOf_id(parent_.id_); }
    inline int32_t const* addressOf_id() const { return parent_.soaMetadata().parametersOf_id().addr_; }
    inline int32_t* addressOf_id() { return parent_.soaMetadata().parametersOf_id().addr_; }
    // XXX size types,
    inline byte_size_type idPitch() const { return (((parent_.nElements_ * sizeof(int32_t) - 1) / ParentClass::alignment) + 1) * ParentClass::alignment; }
    using TypeOf_id = int32_t;
    constexpr static cms::soa::SoAColumnType ColumnTypeOf_id = cms::soa::SoAColumnType::column;
    SoAMetadata& operator=(const SoAMetadata&) = delete;
    SoAMetadata(const SoAMetadata&) = delete;

  private:
    inline SoAMetadata(const SoALayoutTemplate& parent) : parent_(parent) {}
    const SoALayoutTemplate& parent_;
    using ParentClass = SoALayoutTemplate;
  };

  friend SoAMetadata;
  inline const SoAMetadata soaMetadata() const { return SoAMetadata(*this); }
  inline SoAMetadata soaMetadata() { return SoAMetadata(*this); }
  SoALayoutTemplate() : mem_(nullptr), nElements_(0), byteSize_(0), x_(nullptr), y_(nullptr), z_(nullptr), id_(nullptr) {}
  // XXX size types,
  SoALayoutTemplate(std::byte* mem, size_type nElements) : mem_(mem), nElements_(nElements), byteSize_(0) { organizeColumnsFromBuffer(); }

private:
  void organizeColumnsFromBuffer() {
    if constexpr (alignmentEnforcement == cms::soa::AlignmentEnforcement::Enforced)
      if (reinterpret_cast<intptr_t>(mem_) % alignment)
        throw std::out_of_range(
            "In "
            "SoALayoutTemplate"
            "::"
            "SoALayoutTemplate"
            ": misaligned buffer");
    auto curMem = mem_;
    x_ = reinterpret_cast<double*>(curMem);
    curMem += (((nElements_ * sizeof(double) - 1) / alignment) + 1) * alignment;
    if constexpr (alignmentEnforcement == AlignmentEnforcement::Enforced)
      if (reinterpret_cast<intptr_t>(x_) % alignment)
        throw std::out_of_range(
            "In layout constructor: misaligned column: "
            "x");
    y_ = reinterpret_cast<double*>(curMem);
    curMem += (((nElements_ * sizeof(double) - 1) / alignment) + 1) * alignment;
    if constexpr (alignmentEnforcement == AlignmentEnforcement::Enforced)
      if (reinterpret_cast<intptr_t>(y_) % alignment)
        throw std::out_of_range(
            "In layout constructor: misaligned column: "
            "y");
    z_ = reinterpret_cast<double*>(curMem);
    curMem += (((nElements_ * sizeof(double) - 1) / alignment) + 1) * alignment;
    if constexpr (alignmentEnforcement == AlignmentEnforcement::Enforced)
      if (reinterpret_cast<intptr_t>(z_) % alignment)
        throw std::out_of_range(
            "In layout constructor: misaligned column: "
            "z");
    id_ = reinterpret_cast<int32_t*>(curMem);
    curMem += (((nElements_ * sizeof(int32_t) - 1) / alignment) + 1) * alignment;
    if constexpr (alignmentEnforcement == AlignmentEnforcement::Enforced)
      if (reinterpret_cast<intptr_t>(id_) % alignment)
        throw std::out_of_range(
            "In layout constructor: misaligned column: "
            "id");
    byteSize_ = computeDataSize(nElements_);
    if (mem_ + byteSize_ != curMem)
      throw std::out_of_range(
          "In "
          "SoALayoutTemplate"
          "::"
          "SoALayoutTemplate"
          ": unexpected end pointer.");
  }

  // XXX size types,
public:
  SoALayoutTemplate(bool devConstructor, std::byte* mem, size_type nElements) : mem_(mem), nElements_(nElements) {
    auto curMem = mem_;
    x_ = reinterpret_cast<double*>(curMem);
    curMem += (((nElements_ * sizeof(double) - 1) / alignment) + 1) * alignment;
    if constexpr (alignmentEnforcement == AlignmentEnforcement::Enforced)
      if (reinterpret_cast<intptr_t>(x_) % alignment)
        throw std::out_of_range(
            "In layout constructor: misaligned column: "
            "x");
    y_ = reinterpret_cast<double*>(curMem);
    curMem += (((nElements_ * sizeof(double) - 1) / alignment) + 1) * alignment;
    if constexpr (alignmentEnforcement == AlignmentEnforcement::Enforced)
      if (reinterpret_cast<intptr_t>(y_) % alignment)
        throw std::out_of_range(
            "In layout constructor: misaligned column: "
            "y");
    z_ = reinterpret_cast<double*>(curMem);
    curMem += (((nElements_ * sizeof(double) - 1) / alignment) + 1) * alignment;
    if constexpr (alignmentEnforcement == AlignmentEnforcement::Enforced)
      if (reinterpret_cast<intptr_t>(z_) % alignment)
        throw std::out_of_range(
            "In layout constructor: misaligned column: "
            "z");
    id_ = reinterpret_cast<int32_t*>(curMem);
    curMem += (((nElements_ * sizeof(int32_t) - 1) / alignment) + 1) * alignment;
    if constexpr (alignmentEnforcement == AlignmentEnforcement::Enforced)
      if (reinterpret_cast<intptr_t>(id_) % alignment)
        throw std::out_of_range(
            "In layout constructor: misaligned column: "
            "id");
  }

  template <typename T>
  void AllocateAndIoReaToDo(T& onfile) {
    nElements_ = onfile.nElements_;
    std::cout << "AllocateAndIoRead begin" << std::endl;
    auto buffSize = computeDataSize(nElements_);
    //std::cout << "Buffer=" << optionallyOwnedMem_.get() << " Buffer first byte after (alloc) =" << optionallyOwnedMem_.get() + buffSize << std::endl;
    //organizeColumnsFromBuffer();
    memcpy(x_, onfile.x_, sizeof(double) * onfile.nElements_);
    memcpy(y_, onfile.y_, sizeof(double) * onfile.nElements_);
    memcpy(z_, onfile.z_, sizeof(double) * onfile.nElements_);
    memcpy(id_, onfile.id_, sizeof(int32_t) * onfile.nElements_);
    std::cout << "AllocateAndIoRead end" << std::endl;
  }

  template <typename T>
  void AllocateAndIoReadRawToDo(T& onfile) {
    nElements_ = onfile.nElements_;
    std::cout << "AllocateAndIoRead begin" << std::endl;
    auto buffSize = computeDataSize(nElements_);
    //std::cout << "Buffer=" << optionallyOwnedMem_.get() << " Buffer first byte after (alloc) =" << optionallyOwnedMem_.get() + buffSize << std::endl;
    //organizeColumnsFromBuffer();
    memcpy(x_, onfile.x_, sizeof(double) * onfile.nElements_);
    memcpy(y_, onfile.y_, sizeof(double) * onfile.nElements_);
    memcpy(z_, onfile.z_, sizeof(double) * onfile.nElements_);
    memcpy(id_, onfile.id_, sizeof(int32_t) * onfile.nElements_);
    std::cout << "AllocateAndIoRead end" << std::endl;
  }

  template <typename T>
  friend void dump();
  // XXX size types,

private:
  inline void rangeCheck(size_type index) const {
    if constexpr (false) {
      if (index >= nElements_) {
        printf(
            "In "
            "SoALayoutTemplate"
            "::rangeCheck(): index out of range: %zu with nElements: %zu\n",
            index,
            nElements_);
        ((false) ? static_cast<void>(0) : __assert_fail("false", __FILE__, 7, __PRETTY_FUNCTION__));
      }
    }
  }

  std::byte* mem_;
  size_type nElements_;
  // XXX size types,
  byte_size_type byteSize_;
  double* x_ = nullptr;
  double* y_ = nullptr;
  double* z_ = nullptr;
  int32_t* id_ = nullptr;

  // XXX Addition: trivial view:
  // XXX Changed enum class to bare bool + const values in class.
public:
  // XXX size types,
  template <bool RESTRICT_QUALIFY = cms::soa::RestrictQualify::Disabled, bool RANGE_CHECKING = cms::soa::RangeChecking::Disabled>
  using TrivialView = SoAViewTemplate<ALIGNMENT, ALIGNMENT_ENFORCEMENT, RESTRICT_QUALIFY, RANGE_CHECKING>;

  // XXX Addition: streamer
  template <typename T>
  void ROOTReadStreamer(T onfile) {
    auto size = onfile.layout_.soaMetadata().size();
    memcpy(x_, onfile.layout_.x_, size * sizeof(*x_));
    memcpy(y_, onfile.layout_.y_, size * sizeof(*y_));
    memcpy(z_, onfile.layout_.z_, size * sizeof(*z_));
    memcpy(id_, onfile.layout_.id_, size * sizeof(*id_));
  }
};

using XyzIdSoA = SoALayoutTemplate<>;

#include "DataFormats/XyzId/interface/XyzIdSoAView.h"

#endif  // DataFormats_XyzId_interface_XyzIdSoA_h
