// UTF-8 everywhere on Windows: argv, file paths and environment values.
//
// The narrow argv / fopen / std::fstream(std::string) on Windows use the ANSI
// code page (GBK on Chinese systems), while ggml opens files as UTF-8, so a
// non-ASCII path (e.g. a Chinese user name in %LOCALAPPDATA%) used to fail.
// All strings inside the program are UTF-8; these helpers convert at the edges.
#pragma once

#include <cstdio>
#include <filesystem>
#include <string>
#include <system_error>

namespace mintbot {

// Windows: replaces argv with UTF-8 copies of the real (wide) command line. No-op elsewhere.
void use_utf8_argv(int& argc, char**& argv);

FILE* fopen_utf8(const std::string& path, const char* mode);
std::filesystem::path utf8_path(const std::string& path);
std::string getenv_utf8(const char* name);  // empty if unset

// The message of an error code as UTF-8. On Windows ec.message() of a system
// error is in the ANSI code page and localised (GBK on Chinese systems), which
// would make the JSON it ends up in invalid.
std::string error_text(const std::error_code& ec);

}  // namespace mintbot
