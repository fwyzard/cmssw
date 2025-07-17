// TODO: move to HeterogeneousCore/AlpakaInterface, and into alpaka
#pragma once

#include <alpaka/atomic/AtomicUniformCudaHipBuiltIn.hpp>

#include <type_traits>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#if !defined(ALPAKA_HOST_ONLY)

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

namespace alpaka::trait {

  template <typename THierarchy>
  struct AtomicOp<AtomicCas, AtomicUniformCudaHipBuiltIn, unsigned short int, THierarchy> {
    static __device__ auto atomicOp([[maybe_unused]] AtomicUniformCudaHipBuiltIn const& ctx,
                                    unsigned short int* const addr,
                                    unsigned short int const& compare,
                                    unsigned short int const& value) -> unsigned short int {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 700)
      // NVIDIA Volta (sm 7.0) and more recent architectures support atomicCAS with 16-bit unsigned short.
      return ::atomicCAS(addr, compare, value);
#else
      // NVIDIA Pascal (sm 6.x) and older architectures and HIP/ROCm support atomicCAS only with 32-bit unsigned int.
      // Emulate 16-bit CAS using a 32-bit CAS.
      // Note: this requires that also the other half of the 32-bit word is modified only through atomic operations.
      uintptr_t ptr = reinterpret_cast<uintptr_t>(addr);
      // 4-byte aligned address.
      uint32_t* base_addr = reinterpret_cast<uint32_t*>(ptr & ~0x3);
      // True if high 16 bits, false if low 16 bits.
      bool is_upper = (ptr & 0x2) != 0;

      uint32_t old_word, expected_word;
      do {
        old_word = *reinterpret_cast<uint32_t volatile*>(base_addr);
        uint16_t old_half = is_upper ? (old_word >> 16) : (old_word & 0x0000ffff);
        if (old_half != compare) {
          return old_half;
        }

        uint32_t value_word = old_word;
        if (is_upper) {
          value_word = (old_word & 0x0000ffff) | (static_cast<uint32_t>(value) << 16);
        } else {
          value_word = (old_word & 0xffff0000) | static_cast<uint32_t>(value);
        }
        if constexpr (std::is_same_v<THierarchy, hierarchy::Threads>) {
          expected_word = ::atomicCAS_block(base_addr, old_word, value_word);
        } else {
          expected_word = ::atomicCAS(base_addr, old_word, value_word);
        }
      } while (expected_word != old_word);

      return compare;
#endif
    }
  };

}  // namespace alpaka::trait

#endif
#endif
