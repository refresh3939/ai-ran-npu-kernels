#pragma once
#include <acl/acl.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#define CHECK_ACL(call) do { const aclError e_ = (call); if (e_ != ACL_ERROR_NONE) { \
  std::cerr << __FILE__ << ':' << __LINE__ << " aclError:" << e_ << std::endl; \
  std::exit(EXIT_FAILURE); } } while (0)

inline bool ReadFile(const std::string &path, size_t &actual, void *buffer, size_t capacity) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) return false;
  const auto end = in.tellg();
  if (end <= 0 || static_cast<size_t>(end) > capacity) return false;
  actual = static_cast<size_t>(end); in.seekg(0);
  in.read(static_cast<char *>(buffer), static_cast<std::streamsize>(actual));
  return in.good();
}

inline bool WriteFile(const std::string &path, const void *buffer, size_t bytes) {
  if (buffer == nullptr) return false;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(static_cast<const char *>(buffer), static_cast<std::streamsize>(bytes));
  return out.good();
}
