#pragma once

#include <cstddef>
#include <cstdint>



namespace airan::bri_grouped_rhs_adapter {

constexpr uint16_t ABI_VERSION = 1;

#ifndef BRI_GROUPED_NR
#define BRI_GROUPED_NR 64
#endif

#ifndef BRI_GROUPED_NL
#define BRI_GROUPED_NL 16
#endif

constexpr uint32_t N_RE = 23296;
constexpr uint32_t NR = BRI_GROUPED_NR;
constexpr uint32_t NL = BRI_GROUPED_NL;
constexpr uint32_t GROUP_SIZE = 8;
constexpr size_t CANONICAL_ELEMS = static_cast<size_t>(N_RE) * NR * NL;
constexpr size_t GROUPED_ELEMS =
    static_cast<size_t>(N_RE / GROUP_SIZE) * NR * NL;

constexpr size_t CanonicalIndex(uint32_t re, uint32_t rx, uint32_t layer)
{
    return (static_cast<size_t>(re) * NR + rx) * NL + layer;
}

constexpr size_t GroupedIndex(uint32_t re, uint32_t rx)
{
    return (static_cast<size_t>(re / GROUP_SIZE) * NR + rx) * NL +
           (re % GROUP_SIZE);
}

static_assert(N_RE % GROUP_SIZE == 0,
              "BRI schedule must contain complete groups");

}
