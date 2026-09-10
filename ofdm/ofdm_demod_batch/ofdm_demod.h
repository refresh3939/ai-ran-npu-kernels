// ============================================================================
// ofdm_demod.h — OFDM Demodulator 共享头文件
//
// 5G NR mixed-radix 32×64 FFT-2048 解调器
// 目标平台: Ascend 310P3 (dav_m200, __NPU_ARCH__ == 200x)
//
// 数据流: GM int16 IQ → CP removal → unitary DFT-32 → twiddle
//       → unitary DFT-64 → fp16 S4 输出（自然频序，未 fftshift）
// ============================================================================
#pragma once
#include <cstddef>
#include <cstdint>


namespace ofdm_demod_batch {

// ── OFDM 物理层 ─────────────────────────────────────────────────────────────
constexpr uint32_t N_FFT             = 2048;     // FFT 长度 (5G NR μ=1)
constexpr uint32_t N_SYMBOL          = 14;       // 1 slot 内 OFDM sym 数
constexpr uint32_t SYM_STRIDE        = 2192;     // CP(144) + FFT(2048)
constexpr uint32_t SYM0_CP_OFFSET    = 176;      // sym 0 起始 CP 偏移
constexpr uint32_t N_SAMPLE_PER_SLOT = 30720;    // 1 slot 总 IQ sample 数

// ── Mixed-radix 分解 (N = P × Q) ────────────────────────────────────────────
constexpr uint32_t P                 = 32;       // DFT-32 维度
constexpr uint32_t Q                 = 64;       // DFT-64 维度
constexpr uint32_t TILE_PQ           = P * Q;    // = 2048 (单 sym FFT body)

// ── Cube Mmad 切分 ──────────────────────────────────────────────────────────
constexpr uint16_t M_MMAD            = 32;       // Mmad M
constexpr uint16_t N_SUB             = 16;       // 单 sub-cube N
constexpr uint16_t N_SPLITS          = 4;        // N=64 切 4 个 N=16
constexpr uint16_t K_PHASE1          = 32;       // Phase 1 K
constexpr uint16_t K_PHASE3          = 64;       // Phase 3 K
constexpr uint16_t MAX_M_BATCH       = 128;      // 4 symbols × P rows

// ── 多核切分 ────────────────────────────────────────────────────────────────
constexpr uint32_t BLOCK_DIM         = 4;        // 4 AI Core
constexpr uint32_t SYMBOLS_PER_CORE  = 4;        // 14 sym / 4 core 上取整
constexpr uint32_t TILES_PER_BATCH   = (N_SYMBOL + SYMBOLS_PER_CORE - 1) / SYMBOLS_PER_CORE;
constexpr uint32_t BATCH_X_ELEMS     = SYMBOLS_PER_CORE * TILE_PQ;
constexpr uint32_t BATCH_TILE        = SYMBOLS_PER_CORE * P * Q;
constexpr uint32_t BATCH_HALF_TILE   = SYMBOLS_PER_CORE * P * P;

// ── 量化 ────────────────────────────────────────────────────────────────────
constexpr float    INPUT_SCALE       = 1.0f / 256.0f;
constexpr double   SLOT_DURATION_US  = 500.0;  // 5G NR μ=1, 30 kHz SCS
constexpr uint32_t INPUT_GM_INT16_LEN = 65536;
constexpr uint32_t INPUT_INT16_PER_BATCH = 2 * N_SAMPLE_PER_SLOT;
constexpr uint32_t OUTPUT_ELEMS_PER_BATCH = N_SYMBOL * N_FFT;
constexpr uint32_t DEFAULT_BATCH_SIZE = 8;

}  // namespace ofdm_demod_batch
