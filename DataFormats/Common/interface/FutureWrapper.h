#ifndef DataFormats_Common_FutureWrapper_h
#define DataFormats_Common_FutureWrapper_h

#include <future>
#include <chrono>

// simple class used to store an std::shared_future in an edm::Event

template <typename T>
class FutureWrapper {

  FutureWrapper() = default;
  FutureWrapper(FutureWrapper const &) = default;
  FutureWrapper(FutureWrapper &&) = default;
  FutureWrapper & operator= (FutureWrapper const &) = default;

  FutureWrapper(std::shared_future<T> const & f) : m_future(f) { };
  FutureWrapper(std::shared_future<T> && f) : m_future(f) { };
  FutureWrapper(std::future<T> && f) : m_future(f) { };

public:
  T const & get() const {
    return m_future.get();
  }

  bool isValid() const {
    return m_future.valid();
  }

  bool isReady() const {
    return m_future.valid() and m_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
  }


private:
  std::shared_future<T> m_future;

};

#endif // not defined DataFormats_Common_FutureWrapper_h
