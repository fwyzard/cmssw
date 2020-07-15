#ifndef HLTrigger_Tools_interface_Logger_h
#define HLTrigger_Tools_interface_Logger_h

#include <iostream>
#include <sstream>
#include <string>

#include <fmt/printf.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace cms::log {

    // logging levels
    enum class Level {
      debug,
      info,
      warning,
      error,
      system
    };

    // trait to match the logging levels and formatting style to the MessageLogger behaviour
    template <Level level, bool decorate>
    struct LoggerBaseTrait { };

    template <>
    struct LoggerBaseTrait<Level::debug, true> {
      using Type = edm::LogInfo;
    };

    template <>
    struct LoggerBaseTrait<Level::debug, false> {
      using Type = edm::LogVerbatim;
    };

    template <>
    struct LoggerBaseTrait<Level::info, true> {
      using Type = edm::LogInfo;
    };

    template <>
    struct LoggerBaseTrait<Level::info, false> {
      using Type = edm::LogVerbatim;
    };

    template <>
    struct LoggerBaseTrait<Level::warning, true> {
      using Type = edm::LogWarning;
    };

    template <>
    struct LoggerBaseTrait<Level::warning, false> {
      using Type = edm::LogPrint;
    };

    template <>
    struct LoggerBaseTrait<Level::error, true> {
      using Type = edm::LogError;
    };

    template <>
    struct LoggerBaseTrait<Level::error, false> {
      using Type = edm::LogProblem;
    };

    template <>
    struct LoggerBaseTrait<Level::system, true> {
      using Type = edm::LogSystem;
    };

    template <>
    struct LoggerBaseTrait<Level::system, false> {
      using Type = edm::LogAbsolute;
    };

    template <Level level, bool decorate>
    using LoggerBase = typename LoggerBaseTrait<level, decorate>::Type;


    // user interface
    template <Level level, bool decorate>
    class Logger {
    public:
      Logger(std::string const& category) : category_(category) {}

      Logger(std::string&& category) : category_(std::move(category)) {}

      ~Logger() {
        LoggerBase<level, decorate>(category_) << message_.str();
      }

      Logger(Logger const&) = delete;
      Logger(Logger&&) = delete;
      Logger& operator=(Logger const&) = delete;
      Logger& operator=(Logger&&) = delete;

      template <typename T>
      inline Logger& operator<<(T const& t) {
        message_ << t;
        return *this;
      }

      inline Logger& operator<<(std::ostream& (*f)(std::ostream&)) {
        message_ << f;
        return *this;
      }

      inline Logger& operator<<(std::ios_base& (*f)(std::ios_base&)) {
        message_ << f;
        return *this;
      }

      template <typename... Args>
      inline Logger& format(std::string_view fmt, Args const&... args) {
        message_ << fmt::format(fmt, args...);
        return *this;
      }

      template <typename... Args>
      inline Logger& printf(std::string_view fmt, Args const&... args) {
        message_ << fmt::sprintf(fmt, args...);
        return *this;
      }

    private:
      std::string category_;
      std::stringstream message_;
    };

    // concrete types based on the MessageLogger levels
    using system    = Logger<Level::system, true>;
    using absolute  = Logger<Level::system, false>;
    using error     = Logger<Level::error, true>;
    using problem   = Logger<Level::error, false>;
    using warning   = Logger<Level::warning, true>;
    using print     = Logger<Level::warning, false>;
    using info      = Logger<Level::info, true>;
    using verbatim  = Logger<Level::info, false>;

}  // namespace cms::log

#endif  // HLTrigger_Tools_interface_Logger_h
