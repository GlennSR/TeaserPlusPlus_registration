#pragma once
/// Minimal logger – mirrors Python's logging.getLogger() / setup_logging().
/// Each translation unit creates its own Logger instance with __FILE__ as name.

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

enum class LogLevel { DEBUG = 0, INFO = 1, WARNING = 2, ERROR = 3, CRITICAL = 4 };

inline LogLevel log_level_from_string(const std::string& s) {
    if (s == "DEBUG")    return LogLevel::DEBUG;
    if (s == "WARNING")  return LogLevel::WARNING;
    if (s == "ERROR")    return LogLevel::ERROR;
    if (s == "CRITICAL") return LogLevel::CRITICAL;
    return LogLevel::INFO;
}

/// Shared sink that all Logger instances write to.
struct LogSink {
    LogLevel         global_level{LogLevel::INFO};
    std::ofstream    file_stream;
    std::mutex       mtx;

    static LogSink& instance() {
        static LogSink s;
        return s;
    }

    void setup(LogLevel level, const std::string& filename = "", const std::string& filemode = "a") {
        std::lock_guard<std::mutex> lk(mtx);
        global_level = level;
        if (!filename.empty()) {
            auto mode = (filemode == "w") ? std::ios::trunc : std::ios::app;
            file_stream.open(filename, mode);
        }
    }

    void write(LogLevel level, const std::string& name, const std::string& msg) {
        if (level < global_level) return;
        const char* labels[] = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"};
        std::ostringstream oss;
        oss << "[" << labels[static_cast<int>(level)] << "]["
            << name << "] " << msg << "\n";
        std::string line = oss.str();

        std::lock_guard<std::mutex> lk(mtx);
        if (level >= LogLevel::WARNING)
            std::cerr << line;
        else
            std::cout << line;

        if (file_stream.is_open())
            file_stream << line;
    }
};

/// Per-module logger – analogous to Python's logging.getLogger(__name__).
class Logger {
public:
    explicit Logger(std::string name) : name_(std::move(name)) {}

    void debug(const std::string& msg)    { LogSink::instance().write(LogLevel::DEBUG,    name_, msg); }
    void info(const std::string& msg)     { LogSink::instance().write(LogLevel::INFO,     name_, msg); }
    void warning(const std::string& msg)  { LogSink::instance().write(LogLevel::WARNING,  name_, msg); }
    void error(const std::string& msg)    { LogSink::instance().write(LogLevel::ERROR,    name_, msg); }
    void critical(const std::string& msg) { LogSink::instance().write(LogLevel::CRITICAL, name_, msg); }

private:
    std::string name_;
};

/// Mirrors Python's setup_logging(level, filename, filemode).
inline void setup_logging(LogLevel level,
                          const std::string& filename = "",
                          const std::string& filemode = "a") {
    LogSink::instance().setup(level, filename, filemode);
}

/// Convenience: create a named logger (call once per translation unit).
inline Logger get_logger(const std::string& name) { return Logger(name); }
