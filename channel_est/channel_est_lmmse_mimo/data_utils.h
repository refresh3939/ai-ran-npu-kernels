



#pragma once

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>


#define CHECK_ACL(expression)                                                       \
    do {                                                                            \
        const aclError airan_acl_status = (expression);                              \
        if (airan_acl_status != ACL_ERROR_NONE) {                                    \
            std::fprintf(stderr, "[FAIL] %s:%d ACL error %d from %s\n",             \
                         __FILE__, __LINE__, static_cast<int>(airan_acl_status),      \
                         #expression);                                               \
            std::exit(EXIT_FAILURE);                                                 \
        }                                                                            \
    } while (0)




inline bool ReadFile(const std::string &path, size_t &fileSize,
                     void *buffer, size_t expectedSize)
{
    fileSize = 0;
    if ((buffer == nullptr && expectedSize != 0) ||
        expectedSize > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        std::fprintf(stderr, "[FAIL] invalid exact-read request: %s size=%zu\n",
                     path.c_str(), expectedSize);
        return false;
    }

    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input.is_open()) {
        std::fprintf(stderr, "[FAIL] missing/unreadable input: %s (%s)\n",
                     path.c_str(), std::strerror(errno));
        return false;
    }
    const std::streampos end = input.tellg();
    if (end < 0 || static_cast<uintmax_t>(end) != expectedSize) {
        const long long actual = end < 0 ? -1 : static_cast<long long>(end);
        std::fprintf(stderr, "[FAIL] input size mismatch: %s expected=%zu actual=%lld\n",
                     path.c_str(), expectedSize, actual);
        return false;
    }
    input.seekg(0, std::ios::beg);
    if (expectedSize != 0) {
        input.read(static_cast<char *>(buffer), static_cast<std::streamsize>(expectedSize));
        if (!input || static_cast<size_t>(input.gcount()) != expectedSize) {
            std::fprintf(stderr, "[FAIL] short read: %s expected=%zu actual=%lld\n",
                         path.c_str(), expectedSize,
                         static_cast<long long>(input.gcount()));
            return false;
        }
    }
    fileSize = expectedSize;
    return true;
}


inline bool WriteFile(const std::string &path, const void *buffer, size_t size)
{
    if ((buffer == nullptr && size != 0) ||
        size > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        std::fprintf(stderr, "[FAIL] invalid exact-write request: %s size=%zu\n",
                     path.c_str(), size);
        return false;
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
        std::fprintf(stderr, "[FAIL] cannot open output: %s (%s)\n",
                     path.c_str(), std::strerror(errno));
        return false;
    }
    if (size != 0) {
        output.write(static_cast<const char *>(buffer), static_cast<std::streamsize>(size));
    }
    output.flush();
    if (!output) {
        std::fprintf(stderr, "[FAIL] exact write failed: %s size=%zu\n",
                     path.c_str(), size);
        return false;
    }
    output.close();
    if (output.fail()) {
        std::fprintf(stderr, "[FAIL] output close failed: %s\n", path.c_str());
        return false;
    }
    return true;
}
