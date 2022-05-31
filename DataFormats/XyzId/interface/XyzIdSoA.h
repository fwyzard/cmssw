#ifndef DataFormats_XyzId_interface_XyzIdSoA_h
#define DataFormats_XyzId_interface_XyzIdSoA_h

#include "DataFormats/XyzId/interface/SoACommon.h"
#include "DataFormats/XyzId/interface/SoALayout.h"
#include "DataFormats/XyzId/interface/SoAView.h"

#ifdef DEBUG_SOA_CTOR_DTOR
#include <iostream>
#endif

// XXX Addition: forward declaration of trivial view, new size type
// XXX translation of typedef in template for ROOT (cms::soa::byte_size_type = size_t)
template <size_t ALIGNMENT = cms::soa::CacheLineSize::defaultSize,
          bool ALIGNMENT_ENFORCEMENT = cms::soa::AlignmentEnforcement::Relaxed,
          bool RESTRICT_QUALIFY = cms::soa::RestrictQualify::Disabled,
          bool RANGE_CHECKING = cms::soa::RangeChecking::Disabled>
struct SoAViewTemplate;

// XXX new size type.
template <size_t ALIGNMENT = cms::soa::CacheLineSize::defaultSize,
          bool ALIGNMENT_ENFORCEMENT = cms::soa::AlignmentEnforcement::Relaxed>
struct SoALayoutTemplate {
  using self_type = SoALayoutTemplate;

  // XXX size types,
  using size_type = cms::soa::size_type;
  using byte_size_type = cms::soa::byte_size_type;
  using AlignmentEnforcement = cms::soa::AlignmentEnforcement;
  constexpr static byte_size_type defaultAlignment = 128;
  constexpr static byte_size_type alignment = ALIGNMENT;
  constexpr static bool alignmentEnforcement = ALIGNMENT_ENFORCEMENT;
  constexpr static byte_size_type conditionalAlignment =
      alignmentEnforcement == cms::soa::AlignmentEnforcement::Enforced ? alignment : 0;
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
       << offset << " has size " << sizeof(double) * nElements_ << " and padding "
       << (((nElements_ * sizeof(double) - 1) / alignment) + 1) * alignment - (sizeof(double) * nElements_)
       << std ::endl;
    offset += (((nElements_ * sizeof(double) - 1) / alignment) + 1) * alignment;
    os << " Column "
          "y"
          " at offset "
       << offset << " has size " << sizeof(double) * nElements_ << " and padding "
       << (((nElements_ * sizeof(double) - 1) / alignment) + 1) * alignment - (sizeof(double) * nElements_)
       << std ::endl;
    offset += (((nElements_ * sizeof(double) - 1) / alignment) + 1) * alignment;
    os << " Column "
          "z"
          " at offset "
       << offset << " has size " << sizeof(double) * nElements_ << " and padding "
       << (((nElements_ * sizeof(double) - 1) / alignment) + 1) * alignment - (sizeof(double) * nElements_)
       << std ::endl;
    offset += (((nElements_ * sizeof(double) - 1) / alignment) + 1) * alignment;
    os << " Column "
          "id"
          " at offset "
       << offset << " has size " << sizeof(int32_t) * nElements_ << " and padding "
       << (((nElements_ * sizeof(int32_t) - 1) / alignment) + 1) * alignment - (sizeof(int32_t) * nElements_)
       << std ::endl;
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
    inline SoALayoutTemplate cloneToNewAddress(std::byte* addr) const {
      return SoALayoutTemplate(addr, parent_.nElements_);
    }
    typedef cms ::soa ::SoAParameters_ColumnType<cms ::soa ::SoAColumnType ::column>::DataType<double>
        ParametersTypeOf_x;
    inline ParametersTypeOf_x parametersOf_x() const { return ParametersTypeOf_x(parent_.x_); }
    inline double const* addressOf_x() const { return parent_.soaMetadata().parametersOf_x().addr_; }
    inline double* addressOf_x() { return parent_.soaMetadata().parametersOf_x().addr_; }
    // XXX size types,
    inline byte_size_type xPitch() const {
      return (((parent_.nElements_ * sizeof(double) - 1) / ParentClass::alignment) + 1) * ParentClass::alignment;
    }
    using TypeOf_x = double;
    constexpr static cms ::soa ::SoAColumnType ColumnTypeOf_x = cms ::soa ::SoAColumnType ::column;
    typedef cms ::soa ::SoAParameters_ColumnType<cms ::soa ::SoAColumnType ::column>::DataType<double>
        ParametersTypeOf_y;
    inline ParametersTypeOf_y parametersOf_y() const { return ParametersTypeOf_y(parent_.y_); }
    inline double const* addressOf_y() const { return parent_.soaMetadata().parametersOf_y().addr_; }
    inline double* addressOf_y() { return parent_.soaMetadata().parametersOf_y().addr_; }
    // XXX size types,
    inline byte_size_type yPitch() const {
      return (((parent_.nElements_ * sizeof(double) - 1) / ParentClass::alignment) + 1) * ParentClass::alignment;
    }
    using TypeOf_y = double;
    constexpr static cms ::soa ::SoAColumnType ColumnTypeOf_y = cms ::soa ::SoAColumnType ::column;
    typedef cms ::soa ::SoAParameters_ColumnType<cms ::soa ::SoAColumnType ::column>::DataType<double>
        ParametersTypeOf_z;
    inline ParametersTypeOf_z parametersOf_z() const { return ParametersTypeOf_z(parent_.z_); }
    inline double const* addressOf_z() const { return parent_.soaMetadata().parametersOf_z().addr_; }
    inline double* addressOf_z() { return parent_.soaMetadata().parametersOf_z().addr_; }
    // XXX size types,
    inline byte_size_type zPitch() const {
      return (((parent_.nElements_ * sizeof(double) - 1) / ParentClass::alignment) + 1) * ParentClass::alignment;
    }
    using TypeOf_z = double;
    constexpr static cms ::soa ::SoAColumnType ColumnTypeOf_z = cms ::soa ::SoAColumnType ::column;
    typedef cms ::soa ::SoAParameters_ColumnType<cms ::soa ::SoAColumnType ::column>::DataType<int32_t>
        ParametersTypeOf_id;
    inline ParametersTypeOf_id parametersOf_id() const { return ParametersTypeOf_id(parent_.id_); }
    inline int32_t const* addressOf_id() const { return parent_.soaMetadata().parametersOf_id().addr_; }
    inline int32_t* addressOf_id() { return parent_.soaMetadata().parametersOf_id().addr_; }
    // XXX size types,
    inline byte_size_type idPitch() const {
      return (((parent_.nElements_ * sizeof(int32_t) - 1) / ParentClass::alignment) + 1) * ParentClass::alignment;
    }
    using TypeOf_id = int32_t;
    constexpr static cms ::soa ::SoAColumnType ColumnTypeOf_id = cms ::soa ::SoAColumnType ::column;
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
  SoALayoutTemplate()
      : mem_(nullptr), nElements_(0), byteSize_(0), x_(nullptr), y_(nullptr), z_(nullptr), id_(nullptr) {}
  // XXX size types,
  SoALayoutTemplate(std::byte* mem, size_type nElements) : mem_(mem), nElements_(nElements), byteSize_(0) {
    organizeColumnsFromBuffer();
  }

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
  template <size_type ALIGNMENT2 = cms::soa::CacheLineSize::defaultSize,
            bool ALIGNMENT_ENFORCEMENT2 = cms::soa::AlignmentEnforcement::Relaxed,
            bool RESTRICT_QUALIFY2 = cms::soa::RestrictQualify::Disabled,
            bool RANGE_CHECKING2 = cms::soa::RangeChecking::Disabled>
  using TrivialView = SoAViewTemplate<ALIGNMENT2, ALIGNMENT_ENFORCEMENT2, RESTRICT_QUALIFY2, RANGE_CHECKING2>;

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

// XXX Removal of defaults due to forward declaration, switch to bool RESTRICT_QUALIFY and RANGE_CHECKING
// XXX size types,
// XXX translation of typedef in template for ROOT (cms::soa::byte_size_type = size_t)
template <size_t ALIGNMENT, bool ALIGNMENT_ENFORCEMENT, bool RESTRICT_QUALIFY, bool RANGE_CHECKING>
struct SoAViewTemplate {
  using self_type = SoAViewTemplate;
  using layout_type = SoALayoutTemplate<ALIGNMENT, ALIGNMENT_ENFORCEMENT>;

  // XXX szie types,
  using size_type = cms::soa::size_type;
  using byte_size_type = cms::soa::byte_size_type;
  using AlignmentEnforcement = cms::soa::AlignmentEnforcement;
  constexpr static byte_size_type defaultAlignment = cms::soa::CacheLineSize::defaultSize;
  constexpr static byte_size_type alignment = ALIGNMENT;
  constexpr static bool alignmentEnforcement = ALIGNMENT_ENFORCEMENT;
  constexpr static byte_size_type conditionalAlignment =
      alignmentEnforcement == AlignmentEnforcement::Enforced ? alignment : 0;
  // XXX Changed enum class to bare bool + const values in class.
  constexpr static bool restrictQualify = RESTRICT_QUALIFY;
  // XXX Changed enum class to bare bool + const values in class.
  constexpr static bool rangeChecking = RANGE_CHECKING;
  template <cms::soa::SoAColumnType COLUMN_TYPE, class C>
  using SoAValueWithConf = cms::soa::SoAValue<COLUMN_TYPE, C, conditionalAlignment, restrictQualify>;
  template <cms::soa::SoAColumnType COLUMN_TYPE, class C>
  using SoAConstValueWithConf = cms::soa::SoAConstValue<COLUMN_TYPE, C, conditionalAlignment, restrictQualify>;
  struct SoAMetadata {
    friend SoAViewTemplate;
    // XXX size types,
    inline size_type size() const { return parent_.nElements_; }
    using TypeOf_x = typename layout_type ::SoAMetadata::TypeOf_x;
    using ParametersTypeOf_x = typename layout_type ::SoAMetadata::ParametersTypeOf_x;
    constexpr static cms::soa::SoAColumnType ColumnTypeOf_x = layout_type ::SoAMetadata::ColumnTypeOf_x;
    inline auto* addressOf_x() const { return parent_.soaMetadata().parametersOf_x().addr_; };
    inline ParametersTypeOf_x parametersOf_x() const { return parent_.xParameters_; };
    using TypeOf_y = typename layout_type ::SoAMetadata::TypeOf_y;
    using ParametersTypeOf_y = typename layout_type ::SoAMetadata::ParametersTypeOf_y;
    constexpr static cms::soa::SoAColumnType ColumnTypeOf_y = layout_type ::SoAMetadata::ColumnTypeOf_y;
    inline auto* addressOf_y() const { return parent_.soaMetadata().parametersOf_y().addr_; };
    inline ParametersTypeOf_y parametersOf_y() const { return parent_.yParameters_; };
    using TypeOf_z = typename layout_type ::SoAMetadata::TypeOf_z;
    using ParametersTypeOf_z = typename layout_type ::SoAMetadata::ParametersTypeOf_z;
    constexpr static cms::soa::SoAColumnType ColumnTypeOf_z = layout_type ::SoAMetadata::ColumnTypeOf_z;
    inline auto* addressOf_z() const { return parent_.soaMetadata().parametersOf_z().addr_; };
    inline ParametersTypeOf_z parametersOf_z() const { return parent_.zParameters_; };
    using TypeOf_id = typename layout_type ::SoAMetadata::TypeOf_id;
    using ParametersTypeOf_id = typename layout_type ::SoAMetadata::ParametersTypeOf_id;
    constexpr static cms::soa::SoAColumnType ColumnTypeOf_id = layout_type ::SoAMetadata::ColumnTypeOf_id;
    inline auto* addressOf_id() const { return parent_.soaMetadata().parametersOf_id().addr_; };
    inline ParametersTypeOf_id parametersOf_id() const { return parent_.idParameters_; };
    SoAMetadata& operator=(const SoAMetadata&) = delete;
    SoAMetadata(const SoAMetadata&) = delete;

  private:
    inline SoAMetadata(const SoAViewTemplate& parent) : parent_(parent) {}
    const SoAViewTemplate& parent_;
  };
  friend SoAMetadata;
  inline const SoAMetadata soaMetadata() const { return SoAMetadata(*this); }
  inline SoAMetadata soaMetadata() { return SoAMetadata(*this); }
  SoAViewTemplate() {}
  // XXX size types,
  SoAViewTemplate(layout_type& instance_SoALayoutTemplate)
      : nElements_([&]() -> size_type {
          bool set = false;
          // XXX size types,
          size_type ret = 0;
          if (set) {
            if (ret != instance_SoALayoutTemplate.soaMetadata().size())
              throw std ::out_of_range("In constructor by layout: different sizes from layouts.");
          } else {
            ret = instance_SoALayoutTemplate.soaMetadata().size();
            set = true;
          }
          return ret;
        }()),
        xParameters_([&]() -> auto {
          auto params = instance_SoALayoutTemplate.soaMetadata().parametersOf_x();
          if constexpr (alignmentEnforcement == AlignmentEnforcement ::Enforced)
            if (reinterpret_cast<intptr_t>(params.addr_) % alignment)
              throw std ::out_of_range(
                  "In constructor by layout: misaligned column: "
                  "x");
          return params;
        }()),
        yParameters_([&]() -> auto {
          auto params = instance_SoALayoutTemplate.soaMetadata().parametersOf_y();
          if constexpr (alignmentEnforcement == AlignmentEnforcement ::Enforced)
            if (reinterpret_cast<intptr_t>(params.addr_) % alignment)
              throw std ::out_of_range(
                  "In constructor by layout: misaligned column: "
                  "y");
          return params;
        }()),
        zParameters_([&]() -> auto {
          auto params = instance_SoALayoutTemplate.soaMetadata().parametersOf_z();
          if constexpr (alignmentEnforcement == AlignmentEnforcement ::Enforced)
            if (reinterpret_cast<intptr_t>(params.addr_) % alignment)
              throw std ::out_of_range(
                  "In constructor by layout: misaligned column: "
                  "z");
          return params;
        }()),
        idParameters_([&]() -> auto {
          auto params = instance_SoALayoutTemplate.soaMetadata().parametersOf_id();
          if constexpr (alignmentEnforcement == AlignmentEnforcement ::Enforced)
            if (reinterpret_cast<intptr_t>(params.addr_) % alignment)
              throw std ::out_of_range(
                  "In constructor by layout: misaligned column: "
                  "id");
          return params;
        }()) {}
  // XXX size types,
  SoAViewTemplate(size_type nElements,
                  typename SoAMetadata ::ParametersTypeOf_x ::TupleOrPointerType x,
                  typename SoAMetadata ::ParametersTypeOf_y ::TupleOrPointerType y,
                  typename SoAMetadata ::ParametersTypeOf_z ::TupleOrPointerType z,
                  typename SoAMetadata ::ParametersTypeOf_id ::TupleOrPointerType id)
      : nElements_(nElements),
        xParameters_([&]() -> auto {
          if constexpr (alignmentEnforcement == AlignmentEnforcement ::Enforced)
            if (SoAMetadata ::ParametersTypeOf_x ::checkAlignment(x, alignment))
              throw std ::out_of_range(
                  "In constructor by column: misaligned column: "
                  "x");
          return x;
        }()),
        yParameters_([&]() -> auto {
          if constexpr (alignmentEnforcement == AlignmentEnforcement ::Enforced)
            if (SoAMetadata ::ParametersTypeOf_y ::checkAlignment(y, alignment))
              throw std ::out_of_range(
                  "In constructor by column: misaligned column: "
                  "y");
          return y;
        }()),
        zParameters_([&]() -> auto {
          if constexpr (alignmentEnforcement == AlignmentEnforcement ::Enforced)
            if (SoAMetadata ::ParametersTypeOf_z ::checkAlignment(z, alignment))
              throw std ::out_of_range(
                  "In constructor by column: misaligned column: "
                  "z");
          return z;
        }()),
        idParameters_([&]() -> auto {
          if constexpr (alignmentEnforcement == AlignmentEnforcement ::Enforced)
            if (SoAMetadata ::ParametersTypeOf_id ::checkAlignment(id, alignment))
              throw std ::out_of_range(
                  "In constructor by column: misaligned column: "
                  "id");
          return id;
        }()) {}
  struct const_element {
    // XXX size types,
    inline const_element(size_type index,
                         const typename SoAMetadata ::ParametersTypeOf_x x,
                         const typename SoAMetadata ::ParametersTypeOf_y y,
                         const typename SoAMetadata ::ParametersTypeOf_z z,
                         const typename SoAMetadata ::ParametersTypeOf_id id)
        : x_(index, x), y_(index, y), z_(index, z), id_(index, id) {}
    inline typename SoAConstValueWithConf<SoAMetadata ::ColumnTypeOf_x, typename SoAMetadata ::TypeOf_x>::RefToConst x()
        const {
      return x_();
    }
    inline typename SoAConstValueWithConf<SoAMetadata ::ColumnTypeOf_y, typename SoAMetadata ::TypeOf_y>::RefToConst y()
        const {
      return y_();
    }
    inline typename SoAConstValueWithConf<SoAMetadata ::ColumnTypeOf_z, typename SoAMetadata ::TypeOf_z>::RefToConst z()
        const {
      return z_();
    }
    inline typename SoAConstValueWithConf<SoAMetadata ::ColumnTypeOf_id, typename SoAMetadata ::TypeOf_id>::RefToConst
    id() const {
      return id_();
    }

  private:
    const cms::soa::ConstValueTraits<SoAConstValueWithConf<SoAMetadata ::ColumnTypeOf_x, typename SoAMetadata ::TypeOf_x>,
                                     SoAMetadata ::ColumnTypeOf_x>
        x_;
    const cms::soa::ConstValueTraits<SoAConstValueWithConf<SoAMetadata ::ColumnTypeOf_y, typename SoAMetadata ::TypeOf_y>,
                                     SoAMetadata ::ColumnTypeOf_y>
        y_;
    const cms::soa::ConstValueTraits<SoAConstValueWithConf<SoAMetadata ::ColumnTypeOf_z, typename SoAMetadata ::TypeOf_z>,
                                     SoAMetadata ::ColumnTypeOf_z>
        z_;
    const cms::soa::ConstValueTraits<
        SoAConstValueWithConf<SoAMetadata ::ColumnTypeOf_id, typename SoAMetadata ::TypeOf_id>,
        SoAMetadata ::ColumnTypeOf_id>
        id_;
  };
  struct element {
    // XXX size types,
    inline element(size_type index,
                   typename SoAMetadata ::ParametersTypeOf_x x,
                   typename SoAMetadata ::ParametersTypeOf_y y,
                   typename SoAMetadata ::ParametersTypeOf_z z,
                   typename SoAMetadata ::ParametersTypeOf_id id)
        : x(index, x), y(index, y), z(index, z), id(index, id) {}
    inline element& operator=(const element& other) {
      if constexpr (SoAMetadata ::ColumnTypeOf_x != cms ::soa ::SoAColumnType ::scalar)
        x() = other.x();
      if constexpr (SoAMetadata ::ColumnTypeOf_y != cms ::soa ::SoAColumnType ::scalar)
        y() = other.y();
      if constexpr (SoAMetadata ::ColumnTypeOf_z != cms ::soa ::SoAColumnType ::scalar)
        z() = other.z();
      if constexpr (SoAMetadata ::ColumnTypeOf_id != cms ::soa ::SoAColumnType ::scalar)
        id() = other.id();
      return *this;
    }
    SoAValueWithConf<SoAMetadata ::ColumnTypeOf_x, typename SoAMetadata ::TypeOf_x> x;
    SoAValueWithConf<SoAMetadata ::ColumnTypeOf_y, typename SoAMetadata ::TypeOf_y> y;
    SoAValueWithConf<SoAMetadata ::ColumnTypeOf_z, typename SoAMetadata ::TypeOf_z> z;
    SoAValueWithConf<SoAMetadata ::ColumnTypeOf_id, typename SoAMetadata ::TypeOf_id> id;
  };
  // XXX size types,
  inline element operator[](size_type index) {
    if constexpr (rangeChecking == cms::soa::RangeChecking::Enabled) {
      if (index >= nElements_) {
        throw std::out_of_range(
            "Out of range index in "
            "SoAViewTemplate"
            "::operator[]");
      }
    }
    return element(index, xParameters_, yParameters_, zParameters_, idParameters_);
  }
  // XXX size types,
  inline const_element operator[](size_type index) const {
    if constexpr (rangeChecking == cms::soa::RangeChecking::Enabled) {
      if (index >= nElements_) {
        throw std::out_of_range(
            "Out of range index in "
            "SoAViewTemplate"
            "::operator[]");
      }
    }
    return const_element(index, xParameters_, yParameters_, zParameters_, idParameters_);
  }
  inline typename cms ::soa ::SoAAccessors<typename SoAMetadata ::TypeOf_x>::template ColumnType<
      SoAMetadata ::ColumnTypeOf_x>::template AccessType<cms ::soa ::SoAAccessType ::mutableAccess>::NoParamReturnType
  x() {
    return typename cms ::soa ::SoAAccessors<typename SoAMetadata ::TypeOf_x>::template ColumnType<
        SoAMetadata ::ColumnTypeOf_x>::template AccessType<cms ::soa ::SoAAccessType ::mutableAccess>(xParameters_)();
  }
  // XXX size types,
  inline auto& x(size_type index) {
    return typename cms ::soa ::SoAAccessors<typename SoAMetadata ::TypeOf_x>::
        template ColumnType<SoAMetadata ::ColumnTypeOf_x>::template AccessType<cms ::soa ::SoAAccessType ::mutableAccess>(
            xParameters_)(index);
  }
  inline typename cms ::soa ::SoAAccessors<typename SoAMetadata ::TypeOf_y>::template ColumnType<
      SoAMetadata ::ColumnTypeOf_y>::template AccessType<cms ::soa ::SoAAccessType ::mutableAccess>::NoParamReturnType
  y() {
    return typename cms ::soa ::SoAAccessors<typename SoAMetadata ::TypeOf_y>::template ColumnType<
        SoAMetadata ::ColumnTypeOf_y>::template AccessType<cms ::soa ::SoAAccessType ::mutableAccess>(yParameters_)();
  }
  // XXX size types,
  inline auto& y(size_type index) {
    return typename cms ::soa ::SoAAccessors<typename SoAMetadata ::TypeOf_y>::
        template ColumnType<SoAMetadata ::ColumnTypeOf_y>::template AccessType<cms ::soa ::SoAAccessType ::mutableAccess>(
            yParameters_)(index);
  }
  inline typename cms ::soa ::SoAAccessors<typename SoAMetadata ::TypeOf_z>::template ColumnType<
      SoAMetadata ::ColumnTypeOf_z>::template AccessType<cms ::soa ::SoAAccessType ::mutableAccess>::NoParamReturnType
  z() {
    return typename cms ::soa ::SoAAccessors<typename SoAMetadata ::TypeOf_z>::template ColumnType<
        SoAMetadata ::ColumnTypeOf_z>::template AccessType<cms ::soa ::SoAAccessType ::mutableAccess>(zParameters_)();
  }
  // XXX size types,
  inline auto& z(size_type index) {
    return typename cms ::soa ::SoAAccessors<typename SoAMetadata ::TypeOf_z>::
        template ColumnType<SoAMetadata ::ColumnTypeOf_z>::template AccessType<cms ::soa ::SoAAccessType ::mutableAccess>(
            zParameters_)(index);
  }
  inline typename cms ::soa ::SoAAccessors<typename SoAMetadata ::TypeOf_id>::template ColumnType<
      SoAMetadata ::ColumnTypeOf_id>::template AccessType<cms ::soa ::SoAAccessType ::mutableAccess>::NoParamReturnType
  id() {
    return typename cms ::soa ::SoAAccessors<typename SoAMetadata ::TypeOf_id>::template ColumnType<
        SoAMetadata ::ColumnTypeOf_id>::template AccessType<cms ::soa ::SoAAccessType ::mutableAccess>(idParameters_)();
  }
  // XXX size types,
  inline auto& id(size_type index) {
    return typename cms ::soa ::SoAAccessors<typename SoAMetadata ::TypeOf_id>::
        template ColumnType<SoAMetadata ::ColumnTypeOf_id>::template AccessType<
            cms ::soa ::SoAAccessType ::mutableAccess>(idParameters_)(index);
  }
  inline auto x() const {
    return typename cms ::soa ::SoAAccessors<typename SoAMetadata ::TypeOf_x>::template ColumnType<
        SoAMetadata ::ColumnTypeOf_x>::template AccessType<cms ::soa ::SoAAccessType ::constAccess>(xParameters_)();
  }
  // XXX size types,
  inline auto x(size_type index) const {
    return typename cms ::soa ::SoAAccessors<typename SoAMetadata ::TypeOf_x>::template ColumnType<
        SoAMetadata ::ColumnTypeOf_x>::template AccessType<cms ::soa ::SoAAccessType ::constAccess>(xParameters_)(index);
  }
  inline auto y() const {
    return typename cms ::soa ::SoAAccessors<typename SoAMetadata ::TypeOf_y>::template ColumnType<
        SoAMetadata ::ColumnTypeOf_y>::template AccessType<cms ::soa ::SoAAccessType ::constAccess>(yParameters_)();
  }
  // XXX size types,
  inline auto y(size_type index) const {
    return typename cms ::soa ::SoAAccessors<typename SoAMetadata ::TypeOf_y>::template ColumnType<
        SoAMetadata ::ColumnTypeOf_y>::template AccessType<cms ::soa ::SoAAccessType ::constAccess>(yParameters_)(index);
  }
  inline auto z() const {
    return typename cms ::soa ::SoAAccessors<typename SoAMetadata ::TypeOf_z>::template ColumnType<
        SoAMetadata ::ColumnTypeOf_z>::template AccessType<cms ::soa ::SoAAccessType ::constAccess>(zParameters_)();
  }
  // XXX size types,
  inline auto z(size_type index) const {
    return typename cms ::soa ::SoAAccessors<typename SoAMetadata ::TypeOf_z>::template ColumnType<
        SoAMetadata ::ColumnTypeOf_z>::template AccessType<cms ::soa ::SoAAccessType ::constAccess>(zParameters_)(index);
  }
  inline auto id() const {
    return typename cms ::soa ::SoAAccessors<typename SoAMetadata ::TypeOf_id>::template ColumnType<
        SoAMetadata ::ColumnTypeOf_id>::template AccessType<cms ::soa ::SoAAccessType ::constAccess>(idParameters_)();
  }
  // XXX size types,
  inline auto id(size_type index) const {
    return typename cms ::soa ::SoAAccessors<typename SoAMetadata ::TypeOf_id>::
        template ColumnType<SoAMetadata ::ColumnTypeOf_id>::template AccessType<cms ::soa ::SoAAccessType ::constAccess>(
            idParameters_)(index);
  }
  template <typename T>
  friend void dump();
  // XXX size types,
private:
  size_type nElements_ = 0;
  typename SoAMetadata::ParametersTypeOf_x xParameters_;
  typename SoAMetadata::ParametersTypeOf_y yParameters_;
  typename SoAMetadata::ParametersTypeOf_z zParameters_;
  typename SoAMetadata::ParametersTypeOf_id idParameters_;
};

using XyzIdSoA = SoALayoutTemplate<>;

#endif  // DataFormats_XyzId_interface_XyzIdSoA_h
