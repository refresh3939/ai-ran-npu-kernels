// ============================================================================
// 5G NR LDPC decoder kernel — BG1, Z=384, MB=46, KB=22, MAX_DEG=19.
// Target: Ascend 310P1 (dav_m200). Layered Normalized Min-Sum, 4-core CB split.
// Verified: ~7.23--7.29 ms host / 6.785 ms AICore / 5.609 ms Vector
// (3 fixed iterations, 143 CB, CB_GROUP=2, 4 cores, SNR=5 dB).
// ============================================================================

#include "kernel_operator.h"
#include "ldpc_decode.h"

using namespace AscendC;

namespace {

constexpr uint32_t ALIGN16    = 16;
constexpr uint32_t Z          = airan::LDPC_Z;
constexpr uint32_t MAX_DEG    = airan::LDPC_MAX_DEG;
constexpr uint32_t MAX_ITER   = airan::LDPC_MAX_ITER;
constexpr uint32_t MB         = airan::LDPC_MB;
constexpr uint32_t C_NUM      = airan::LDPC_C_NUM;
constexpr uint32_t BLOCK_DIM  = 4;
constexpr uint32_t CB_GROUP   = 2;
constexpr uint32_t ROW_GROUP  = 2;

constexpr uint32_t DEG_PADDED  = ((MB     + ALIGN16 - 1) / ALIGN16) * ALIGN16;

constexpr uint32_t UB_LAM_BYTES      = CB_GROUP * airan::LDPC_NFULL * Z * sizeof(half);
constexpr uint32_t UB_DEG_Z_BYTES    = CB_GROUP * MAX_DEG * Z * sizeof(half);
constexpr uint32_t UB_CONST_Z_BYTES  = Z * sizeof(half);
constexpr uint32_t UB_GROUP_Z_BYTES  = ROW_GROUP * CB_GROUP * Z * sizeof(half);
constexpr uint32_t UB_DEG_BYTES      = DEG_PADDED  * sizeof(int16_t);
// The mask is consumed immediately for each CB lane, so lanes reuse it.
constexpr uint32_t UB_MASK_BYTES     = 3 * MAX_DEG * 32;
constexpr uint32_t UB_PREV_ROW_BYTES = CB_GROUP * MAX_DEG * Z * sizeof(int8_t);

// Adjacent rows in these pairs have equal degree and no common block-column,
// so their layered updates commute and may share one Vector instruction stream.
__aicore__ inline bool FuseNextRow(uint32_t br)
{
    return br == 16 || br == 20 || br == 22 || br == 25 || br == 28 ||
           br == 30 || br == 32 || br == 34 || br == 38 || br == 43;
}

// CyclicShift: dst[i] = src[(i + sh) mod Z].
__aicore__ inline void CyclicShift(const LocalTensor<half>& dst,
                                   const LocalTensor<half>& src,
                                   uint32_t sh)
{
    if (sh == 0) {
        DataCopy(dst, src, (uint32_t)Z);
    } else if ((sh & 15) == 0) {
        const uint32_t head = Z - sh;
        DataCopy(dst,       src[sh], head);
        DataCopy(dst[head], src,     sh);
    } else {
        const uint32_t head = Z - sh;
        Adds(dst,       src[sh], (half)0.0f, (int32_t)head);
        Adds(dst[head], src,     (half)0.0f, (int32_t)sh);
    }
}

// Both tensors use [item][cb_lane][z], so one instruction stream rotates both
// lanes for aligned MTE copies. Arbitrary shifts retain the proven per-lane
// Vector path: the attempted Level-0 continuous-mask variant was not bit-exact.
__aicore__ inline void CyclicShiftInterleaved(const LocalTensor<half>& dst,
                                              const LocalTensor<half>& src,
                                              uint32_t sh,
                                              uint32_t batch)
{
    if (batch == 1) {
        CyclicShift(dst, src, sh);
    } else if ((sh & 15) == 0) {
        const uint32_t head = Z - sh;
        DataCopyParams p;
        p.blockCount = (uint16_t)batch;
        p.blockLen = (uint16_t)(head / ALIGN16);
        p.srcStride = (uint16_t)((Z - head) / ALIGN16);
        p.dstStride = (uint16_t)((Z - head) / ALIGN16);
        DataCopy(dst, src[sh], p);
        if (sh != 0) {
            p.blockLen = (uint16_t)(sh / ALIGN16);
            p.srcStride = (uint16_t)((Z - sh) / ALIGN16);
            p.dstStride = (uint16_t)((Z - sh) / ALIGN16);
            DataCopy(dst[head], src, p);
        }
    } else {
        for (uint32_t b = 0; b < batch; ++b) {
            CyclicShift(dst[b * Z], src[b * Z], sh);
        }
    }
}

// Hot-loop specializations. Graph metadata is partitioned by shift class once
// at kernel entry, so these paths avoid per-edge zero/alignment dispatch.
template <uint32_t BATCH>
__aicore__ inline void CyclicShiftZero(const LocalTensor<half>& dst,
                                       const LocalTensor<half>& src)
{
    DataCopy(dst, src, BATCH * Z);
}

template <uint32_t BATCH>
__aicore__ inline void CyclicShiftAligned(const LocalTensor<half>& dst,
                                          const LocalTensor<half>& src,
                                          uint32_t sh)
{
    const uint32_t head = Z - sh;
    if constexpr (BATCH == 1) {
        DataCopy(dst,       src[sh], head);
        DataCopy(dst[head], src,     sh);
    } else {
        DataCopyParams p;
        p.blockCount = (uint16_t)BATCH;
        p.blockLen = (uint16_t)(head / ALIGN16);
        p.srcStride = (uint16_t)((Z - head) / ALIGN16);
        p.dstStride = (uint16_t)((Z - head) / ALIGN16);
        DataCopy(dst, src[sh], p);
        p.blockLen = (uint16_t)(sh / ALIGN16);
        p.srcStride = (uint16_t)((Z - sh) / ALIGN16);
        p.dstStride = (uint16_t)((Z - sh) / ALIGN16);
        DataCopy(dst[head], src, p);
    }
}

template <uint32_t BATCH>
__aicore__ inline void CyclicShiftVector(const LocalTensor<half>& dst,
                                         const LocalTensor<half>& src,
                                         uint32_t sh)
{
    const uint32_t head = Z - sh;
    #pragma unroll 2
    for (uint32_t b = 0; b < BATCH; ++b) {
        const uint32_t off = b * Z;
        Adds(dst[off],        src[off + sh], (half)0.0f, (int32_t)head);
        Adds(dst[off + head], src[off],      (half)0.0f, (int32_t)sh);
    }
}

template <uint32_t BATCH>
__aicore__ inline void CyclicShiftDynamic(const LocalTensor<half>& dst,
                                          const LocalTensor<half>& src,
                                          uint32_t sh)
{
    if (sh == 0) {
        CyclicShiftZero<BATCH>(dst, src);
    } else if ((sh & 15) == 0) {
        CyclicShiftAligned<BATCH>(dst, src, sh);
    } else {
        CyclicShiftVector<BATCH>(dst, src, sh);
    }
}

// BATCH and ROWS are hot-path invariants.  Keeping them in the type removes
// per-edge CB/row loop control and lets the compiler fold all lane strides.
template <uint32_t BATCH, uint32_t ROWS>
__aicore__ inline void GatherShiftDispatch(
    const LocalTensor<half>& dst,
    const LocalTensor<half>& src,
    const uint32_t* graph_meta,
    const uint16_t* row_shift_split,
    uint32_t br,
    uint32_t pack_off,
    uint32_t deg)
{
    #pragma unroll 2
    for (uint32_t rr = 0; rr < ROWS; ++rr) {
        const uint32_t row_pack = pack_off + rr * MAX_DEG;
        const uint32_t split = row_shift_split[br + rr];
        const uint32_t zero_end = split & 0xffu;
        const uint32_t aligned_end = split >> 8;
        #pragma unroll 4
        for (uint32_t kk = 0; kk < zero_end; ++kk) {
            const uint32_t bc = graph_meta[row_pack + kk] & 0xffffu;
            const uint32_t woff = (kk * ROWS + rr) * BATCH * Z;
            CyclicShiftZero<BATCH>(dst[woff], src[bc * BATCH * Z]);
        }
        #pragma unroll 4
        for (uint32_t kk = zero_end; kk < aligned_end; ++kk) {
            const uint32_t meta = graph_meta[row_pack + kk];
            const uint32_t bc = meta & 0xffffu;
            const uint32_t woff = (kk * ROWS + rr) * BATCH * Z;
            CyclicShiftAligned<BATCH>(dst[woff], src[bc * BATCH * Z],
                                      meta >> 16);
        }
        #pragma unroll 4
        for (uint32_t kk = aligned_end; kk < deg; ++kk) {
            const uint32_t meta = graph_meta[row_pack + kk];
            const uint32_t bc = meta & 0xffffu;
            const uint32_t woff = (kk * ROWS + rr) * BATCH * Z;
            CyclicShiftVector<BATCH>(dst[woff], src[bc * BATCH * Z],
                                     meta >> 16);
        }
    }
}

// The first iteration starts in public variable coordinates, whereas later
// iterations start at the last incident edge coordinate.  Its edge ordering
// follows the steady-state plan for message-slot stability, so shift class is
// selected per edge only on this one cold iteration.
template <uint32_t BATCH, uint32_t ROWS>
__aicore__ inline void GatherShiftDynamic(
    const LocalTensor<half>& dst,
    const LocalTensor<half>& src,
    const uint32_t* graph_meta,
    uint32_t pack_off,
    uint32_t deg)
{
    #pragma unroll 2
    for (uint32_t rr = 0; rr < ROWS; ++rr) {
        const uint32_t row_pack = pack_off + rr * MAX_DEG;
        #pragma unroll 4
        for (uint32_t kk = 0; kk < deg; ++kk) {
            const uint32_t meta = graph_meta[row_pack + kk];
            const uint32_t bc = meta & 0xffffu;
            const uint32_t woff = (kk * ROWS + rr) * BATCH * Z;
            CyclicShiftDynamic<BATCH>(dst[woff], src[bc * BATCH * Z],
                                      meta >> 16);
        }
    }
}

// ext is already expressed in this edge's check coordinate.  Make that the
// resident coordinate for the variable column, replacing inverse-shift
// scatter with one contiguous UB-to-UB copy.
template <uint32_t BATCH, uint32_t ROWS>
__aicore__ inline void ScatterResident(
    const LocalTensor<half>& dst,
    const LocalTensor<half>& src,
    const uint32_t* graph_meta,
    uint32_t pack_off,
    uint32_t deg)
{
    #pragma unroll 2
    for (uint32_t rr = 0; rr < ROWS; ++rr) {
        const uint32_t row_pack = pack_off + rr * MAX_DEG;
        #pragma unroll 4
        for (uint32_t kk = 0; kk < deg; ++kk) {
            const uint32_t bc = graph_meta[row_pack + kk] & 0xffffu;
            const uint32_t woff = (kk * ROWS + rr) * BATCH * Z;
            CyclicShiftZero<BATCH>(dst[bc * BATCH * Z], src[woff]);
        }
    }
}

// BG1 has only nine check degrees.  Specializing the reduction removes the
// dynamic edge loop and fixes the broadcast repeat count at compile time.
template <uint32_t DEG>
__aicore__ inline void CheckNodeDegree(
    const LocalTensor<half>& sign_prod,
    const LocalTensor<half>& sgn,
    const LocalTensor<half>& min1,
    const LocalTensor<half>& min2,
    const LocalTensor<half>& magnitude,
    const LocalTensor<uint8_t>& row_mask,
    uint32_t row_lanes,
    uint8_t lane_reps,
    const BinaryRepeatParams& flat_bin_p,
    const UnaryRepeatParams& alpha_p,
    const BinaryRepeatParams& lane_cmp_p,
    const BinaryRepeatParams& lane_sel_p,
    const BinaryRepeatParams& lane_scale_p)
{
    constexpr uint32_t MASK_Q_STRIDE = MAX_DEG * 32;
    const uint32_t edge_stride = row_lanes * Z;
    Mul<half, false>(sign_prod, sgn, sgn[edge_stride],
                     (uint64_t)128, lane_reps, flat_bin_p);
    Min<half, false>(min1, magnitude, magnitude[edge_stride],
                     (uint64_t)128, lane_reps, flat_bin_p);
    Max<half, false>(min2, magnitude, magnitude[edge_stride],
                     (uint64_t)128, lane_reps, flat_bin_p);
    #pragma unroll
    for (uint32_t kk = 2; kk < DEG; ++kk) {
        const uint32_t off = kk * edge_stride;
        Min<half, false>(min2, min2, magnitude[off],
                         (uint64_t)128, lane_reps, flat_bin_p);
        Max<half, false>(min2, min2, min1,
                         (uint64_t)128, lane_reps, flat_bin_p);
        Min<half, false>(min1, min1, magnitude[off],
                         (uint64_t)128, lane_reps, flat_bin_p);
        Mul<half, false>(sign_prod, sign_prod, sgn[off],
                         (uint64_t)128, lane_reps, flat_bin_p);
    }
    Muls<half, false>(sign_prod, sign_prod, (half)airan::NMS_ALPHA_F,
                      (uint64_t)128, lane_reps, alpha_p);

    #pragma unroll 4
    for (uint32_t lane = 0; lane < row_lanes; ++lane) {
        #pragma unroll
        for (uint32_t q = 0, m = 0; q < Z; q += 128, m += MASK_Q_STRIDE) {
            const uint32_t off = lane * Z + q;
            Compare(row_mask[m], magnitude[off], min1[off], CMPMODE::EQ,
                    (uint64_t)128, (uint8_t)DEG, lane_cmp_p);
            Select(magnitude[off], row_mask[m], min2[off], min1[off],
                   SELMODE::VSEL_TENSOR_TENSOR_MODE,
                   (uint64_t)128, (uint8_t)DEG, lane_sel_p);
            Mul<half, false>(sgn[off], sign_prod[off], sgn[off],
                             (uint64_t)128, (uint8_t)DEG, lane_scale_p);
        }
    }
}

// Restore one moving-coordinate posterior column at final egress.
__aicore__ inline void RotateColumnInPlace(const LocalTensor<half>& column,
                                           const LocalTensor<half>& scratch,
                                           uint32_t sh,
                                           uint32_t batch)
{
    if (sh == 0) return;
    CyclicShiftInterleaved(scratch, column, sh, batch);
    Adds(column, scratch, (half)0.0f, (int32_t)(batch * Z));
}

}  // namespace

extern "C" __global__ __aicore__ void ldpc_decode_kernel(
    GM_ADDR lam_in_gm,
    GM_ADDR packed_bc_gm,
    GM_ADDR packed_s_gm,
    GM_ADDR degrees_gm,
    GM_ADDR edge_offsets_gm,
    GM_ADDR prev_msg_gm,
    GM_ADDR decoded_bits_gm,
    GM_ADDR lam_out_gm,
    GM_ADDR lam_scratch_gm)
{
    const uint32_t aiv_id = (uint32_t)GetBlockIdx() ^ 2U;

    GlobalTensor<int16_t> lamInG, packedBcG, packedSG, degreesG, lamOutG, lamScratchG;
    GlobalTensor<int8_t>  prevMsgG, bitsOutG;

    lamInG     .SetGlobalBuffer((__gm__ int16_t*)lam_in_gm,        (size_t)C_NUM * airan::LAM_ELEMS_PER_CB);
    packedBcG  .SetGlobalBuffer((__gm__ int16_t*)packed_bc_gm,     airan::PACKED_ELEMS_TOTAL);
    packedSG   .SetGlobalBuffer((__gm__ int16_t*)packed_s_gm,      airan::PACKED_ELEMS_TOTAL);
    degreesG   .SetGlobalBuffer((__gm__ int16_t*)degrees_gm,       DEG_PADDED);
    prevMsgG   .SetGlobalBuffer((__gm__ int8_t* )prev_msg_gm,      (size_t)C_NUM * airan::PREV_ELEMS_PER_CB);
    bitsOutG   .SetGlobalBuffer((__gm__ int8_t* )decoded_bits_gm,  (size_t)C_NUM * airan::LDPC_K);
    lamOutG    .SetGlobalBuffer((__gm__ int16_t*)lam_out_gm,       (size_t)C_NUM * airan::LAM_ELEMS_PER_CB);
    lamScratchG.SetGlobalBuffer((__gm__ int16_t*)lam_scratch_gm,   (size_t)4 * airan::LAM_ELEMS_PER_CB);

    TPipe pipe;

    TBuf<TPosition::VECCALC> bufPackedBc, bufPackedS, bufDeg;
    TBuf<TPosition::VECCALC> bufPosOne, bufNegOne;
    TBuf<TPosition::VECCALC> bufLamPadH;
    TBuf<TPosition::VECCALC> bufPrevRow;
    TBuf<TPosition::VECCALC> bufGath, bufSgn, bufNewMsg;
    TBuf<TPosition::VECCALC> bufMin1, bufMin2, bufSignProd;
    TBuf<TPosition::VECCALC> bufMask;

    constexpr uint32_t PACKED_PADDED   = ((airan::PACKED_ELEMS_TOTAL + ALIGN16 - 1) / ALIGN16) * ALIGN16;
    constexpr uint32_t UB_PACKED_BYTES = PACKED_PADDED * sizeof(int16_t);

    pipe.InitBuffer(bufPackedBc, UB_PACKED_BYTES);
    pipe.InitBuffer(bufPackedS,  UB_PACKED_BYTES);
    pipe.InitBuffer(bufDeg,      UB_DEG_BYTES);
    pipe.InitBuffer(bufPosOne,   UB_CONST_Z_BYTES);
    pipe.InitBuffer(bufNegOne,   UB_CONST_Z_BYTES);
    pipe.InitBuffer(bufLamPadH,  UB_LAM_BYTES);
    pipe.InitBuffer(bufPrevRow,  2 * UB_PREV_ROW_BYTES);
    pipe.InitBuffer(bufGath,     UB_DEG_Z_BYTES);
    pipe.InitBuffer(bufSgn,      UB_DEG_Z_BYTES);
    pipe.InitBuffer(bufNewMsg,   UB_DEG_Z_BYTES);
    pipe.InitBuffer(bufMin1,     UB_GROUP_Z_BYTES);
    pipe.InitBuffer(bufMin2,     UB_GROUP_Z_BYTES);
    pipe.InitBuffer(bufSignProd, UB_GROUP_Z_BYTES);
    pipe.InitBuffer(bufMask,     UB_MASK_BYTES);

    auto packed_bc = bufPackedBc.Get<int16_t>();
    auto packed_s  = bufPackedS .Get<int16_t>();
    auto deg_ub    = bufDeg     .Get<int16_t>();
    auto pos_one   = bufPosOne  .Get<half>();
    auto neg_one   = bufNegOne  .Get<half>();

    auto lam_pad     = bufLamPadH.Get<half>();
    auto lam_pad_i16 = bufLamPadH.Get<int16_t>();

    auto prev_row_i8 = bufPrevRow.Get<int8_t>();

    // Gathered values become extrinsic values in place after subtracting the
    // old row message.  This lifetime alias removes one MAX_DEG-sized buffer.
    auto gath_h      = bufGath  .Get<half>();
    auto ext         = bufGath  .Get<half>();
    auto bits_ub     = bufGath  .Get<int8_t>();     // alias: ext dies at S6

    auto sgn         = bufSgn   .Get<half>();
    auto new_msg_h   = bufNewMsg.Get<half>();
    auto prev_h      = bufNewMsg.Get<half>();

    auto min1        = bufMin1   .Get<half>();
    auto min2        = bufMin2   .Get<half>();
    auto sign_prod   = bufSignProd.Get<half>();
    auto row_mask    = bufMask.Get<uint8_t>();

    // dav_m200 cannot safely dereference large global constexpr tables from
    // device code.  Stage the small fixed graph once through UB, then keep a
    // packed scalar-stack views for the hot loop: [shift:16 | block_col:16].
    // A posterior column remains in the coordinate of its most recently
    // processed edge.  Iteration zero therefore has a distinct delta-shift
    // plan; iterations one and two share the cyclic steady-state plan.
    uint8_t row_degree[MB];
    uint16_t last_shift[airan::LDPC_NFULL] = {};
    // Steady-state plan: low byte ends zero shifts, high byte aligned shifts.
    uint16_t row_shift_split[MB];
    uint32_t graph_meta_first[airan::PACKED_ELEMS_TOTAL];
    uint32_t graph_meta_steady[airan::PACKED_ELEMS_TOTAL];

    DataCopy(packed_bc, packedBcG, PACKED_PADDED);
    DataCopy(packed_s,  packedSG,  PACKED_PADDED);
    DataCopy(deg_ub,    degreesG,  DEG_PADDED);

    Duplicate<half>(pos_one,  (half) 1.0f,                       (int32_t)Z);
    Duplicate<half>(neg_one,  (half)-1.0f,                       (int32_t)Z);

    SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);

    for (uint32_t br = 0; br < MB; ++br) {
        row_degree[br] = (uint8_t)deg_ub.GetValue(br);
        const uint32_t base = br * MAX_DEG;
        for (uint32_t kk = 0; kk < row_degree[br]; ++kk) {
            const uint32_t bc =
                (uint32_t)(uint16_t)packed_bc.GetValue(base + kk);
            last_shift[bc] =
                (uint16_t)packed_s.GetValue(base + kk);
        }
    }

    uint16_t current_first[airan::LDPC_NFULL] = {};
    uint16_t current_steady[airan::LDPC_NFULL];
    for (uint32_t bc = 0; bc < airan::LDPC_NFULL; ++bc) {
        current_steady[bc] = last_shift[bc];
    }
    for (uint32_t br = 0; br < MB; ++br) {
        const uint32_t base = br * MAX_DEG;
        const uint32_t deg = row_degree[br];
        uint64_t row_meta[MAX_DEG];
        for (uint32_t kk = 0; kk < deg; ++kk) {
            const uint32_t bc =
                (uint32_t)(uint16_t)packed_bc.GetValue(base + kk);
            const uint32_t sh =
                (uint32_t)(uint16_t)packed_s.GetValue(base + kk);
            const uint32_t first_delta =
                (sh + Z - current_first[bc]) % Z;
            const uint32_t steady_delta =
                (sh + Z - current_steady[bc]) % Z;
            current_first[bc] = (uint16_t)sh;
            current_steady[bc] = (uint16_t)sh;
            const uint32_t first = bc | (first_delta << 16);
            const uint32_t steady = bc | (steady_delta << 16);
            row_meta[kk] = (uint64_t)first | ((uint64_t)steady << 32);
        }

        uint32_t out = 0;
        for (uint32_t kk = 0; kk < deg; ++kk) {
            const uint32_t steady = (uint32_t)(row_meta[kk] >> 32);
            if ((steady >> 16) == 0) {
                graph_meta_first[base + out] = (uint32_t)row_meta[kk];
                graph_meta_steady[base + out++] = steady;
            }
        }
        const uint32_t zero_end = out;
        for (uint32_t kk = 0; kk < deg; ++kk) {
            const uint32_t steady = (uint32_t)(row_meta[kk] >> 32);
            const uint32_t sh = steady >> 16;
            if (sh != 0 && (sh & 15) == 0) {
                graph_meta_first[base + out] = (uint32_t)row_meta[kk];
                graph_meta_steady[base + out++] = steady;
            }
        }
        const uint32_t aligned_end = out;
        for (uint32_t kk = 0; kk < deg; ++kk) {
            const uint32_t steady = (uint32_t)(row_meta[kk] >> 32);
            if (((steady >> 16) & 15) != 0) {
                graph_meta_first[base + out] = (uint32_t)row_meta[kk];
                graph_meta_steady[base + out++] = steady;
            }
        }
        row_shift_split[br] = (uint16_t)(zero_end | (aligned_end << 8));
    }

    constexpr uint8_t  Z_BLK_STRIDE  = (uint8_t)(Z / ALIGN16);
    constexpr uint8_t  Z_REPS        = (uint8_t)(Z / 128);

    // All repeat descriptors are graph invariants.  Construct them once per
    // kernel instead of once per check row (46 rows x 3 iterations x ~36 CBs).
    const UnaryRepeatParams  cast_i8_to_h_p {1, 1, 8, 4};
    const BinaryRepeatParams flat_bin_p     {1, 1, 1, 8, 8, 8};
    const UnaryRepeatParams  flat_abs_p     {1, 1, 8, 8};
    const UnaryRepeatParams  sign_cmp_p     {1, 1, 2, 8};
    const BinaryRepeatParams sign_sel_p     {1, 1, 1, 8, 0, 0};
    const UnaryRepeatParams  alpha_p        {1, 1, 8, 8};
    const BinaryRepeatParams msg_cast_p     {1, 1, 1, 4, 8, 8};

    // Two independent CBs share one instruction stream.  Check rows retain
    // their layered order; only the CB dimension is batched.
    constexpr uint16_t PREV_EDGE_BLOCKS = (uint16_t)(Z * sizeof(int8_t) / 32);
    const uint32_t cb_begin = (C_NUM * aiv_id) / BLOCK_DIM;
    const uint32_t cb_end   = (C_NUM * (aiv_id + 1)) / BLOCK_DIM;
    for (uint32_t cb_base = cb_begin; cb_base < cb_end; cb_base += CB_GROUP) {
        const uint32_t batch = ((cb_end - cb_base) < CB_GROUP) ? (cb_end - cb_base) : CB_GROUP;
        const uint32_t lam_elems = batch * (uint32_t)airan::LAM_ELEMS_PER_CB;
        const size_t cb_lam_base = (size_t)cb_base * airan::LAM_ELEMS_PER_CB;

        // Convert CB-major GM input to [block_col][cb_lane][z] once.  This
        // makes every later cyclic shift naturally batchable across CB lanes.
        if (batch == 1) {
            DataCopy(lam_pad_i16, lamInG[cb_lam_base], lam_elems);
        } else {
            DataCopyParams load_lam;
            load_lam.blockCount = (uint16_t)batch;
            load_lam.blockLen = (uint16_t)(Z * sizeof(int16_t) / 32);
            load_lam.srcStride = (uint16_t)((airan::LAM_ELEMS_PER_CB - Z) * sizeof(int16_t) / 32);
            load_lam.dstStride = 0;
            for (uint32_t bc = 0; bc < airan::LDPC_NFULL; ++bc) {
                DataCopy(lam_pad_i16[bc * batch * Z],
                         lamInG[cb_lam_base + bc * Z], load_lam);
            }
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(lam_pad, lam_pad_i16, RoundMode::CAST_NONE, (int32_t)lam_elems);

        // The CB group is fixed here; row-group-dependent strides are formed
        // below only once per fused/single row group.
        DataCopyParams load_prev;
        load_prev.blockLen = PREV_EDGE_BLOCKS;
        load_prev.srcStride = 0;
        DataCopyParams store_prev;
        store_prev.blockLen = PREV_EDGE_BLOCKS;
        store_prev.dstStride = 0;

        bool store_pending[2] = {false, false};
        for (uint32_t iter = 0; iter < MAX_ITER; ++iter) {
            uint32_t ub_off = 0;
            uint32_t pack_off = 0;
            for (uint32_t br = 0, group_seq = 0; br < MB; ++group_seq) {
                const uint32_t row_count = FuseNextRow(br) ? ROW_GROUP : 1;
                const int32_t deg = (int32_t)row_degree[br];
                const uint32_t edge_elems = (uint32_t)deg * Z;
                const uint32_t row_lanes = row_count * batch;
                const uint32_t work_elems = row_lanes * edge_elems;
                const uint8_t edge_reps = (uint8_t)(row_lanes * (uint32_t)deg * Z_REPS);
                const uint8_t lane_reps = (uint8_t)(row_lanes * Z_REPS);
                const uint8_t row_group_blk_stride = (uint8_t)(row_lanes * Z_BLK_STRIDE);
                const BinaryRepeatParams lane_cmp_p {1, 1, 1, 2, row_group_blk_stride, 0};
                const BinaryRepeatParams lane_sel_p {1, 1, 1, row_group_blk_stride, 0, 0};
                const BinaryRepeatParams lane_scale_p {
                    1, 1, 1, row_group_blk_stride, 0, row_group_blk_stride};
                const uint32_t row_slot = group_seq & 1;
                auto prev_i8 = prev_row_i8[row_slot * UB_PREV_ROW_BYTES];
                // A slot is reused only after one complete intervening row, so
                // the previous MTE3 store overlaps that row's Vector work.
                if (store_pending[row_slot]) {
                    if (row_slot == 0) {
                        WaitFlag<HardEvent::MTE3_V>(EVENT_ID4);
                    } else {
                        WaitFlag<HardEvent::MTE3_V>(EVENT_ID5);
                    }
                    store_pending[row_slot] = false;
                }

                // T1: iteration 0 has zero old messages.  Later iterations load
                // two CB-major GM rows into edge-major UB with one 2D copy/CB.
                if (iter == 0) {
                    Duplicate<half>(prev_h, (half)0.0f, (int32_t)work_elems);
                } else {
                    load_prev.blockCount = (uint16_t)deg;
                    load_prev.dstStride =
                        (uint16_t)((row_lanes - 1) * PREV_EDGE_BLOCKS);
                    #pragma unroll 2
                    for (uint32_t rr = 0; rr < row_count; ++rr) {
                        #pragma unroll 2
                        for (uint32_t b = 0; b < batch; ++b) {
                            const size_t gm_off =
                                (size_t)(cb_base + b) * airan::PREV_ELEMS_PER_CB +
                                ub_off + rr * edge_elems;
                            DataCopy(prev_i8[(rr * batch + b) * Z],
                                     prevMsgG[gm_off], load_prev);
                        }
                    }
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                    Cast<half, int8_t, false>(prev_h, prev_i8, RoundMode::CAST_NONE,
                                              (uint64_t)128, edge_reps, cast_i8_to_h_p);
                }

                // T2: [edge][row][cb][z] keeps independent rows/CBs adjacent
                // under one repeat descriptor. Each row's edges are already
                // partitioned by shift class, so no per-edge dispatch remains.
                if (iter == 0) {
                    if (batch == 2) {
                        if (row_count == 2) {
                            GatherShiftDynamic<2, 2>(gath_h, lam_pad,
                                graph_meta_first, pack_off, deg);
                        } else {
                            GatherShiftDynamic<2, 1>(gath_h, lam_pad,
                                graph_meta_first, pack_off, deg);
                        }
                    } else if (row_count == 2) {
                        GatherShiftDynamic<1, 2>(gath_h, lam_pad,
                            graph_meta_first, pack_off, deg);
                    } else {
                        GatherShiftDynamic<1, 1>(gath_h, lam_pad,
                            graph_meta_first, pack_off, deg);
                    }
                } else if (batch == 2) {
                    if (row_count == 2) {
                        GatherShiftDispatch<2, 2>(gath_h, lam_pad,
                            graph_meta_steady, row_shift_split,
                            br, pack_off, deg);
                    } else {
                        GatherShiftDispatch<2, 1>(gath_h, lam_pad,
                            graph_meta_steady, row_shift_split,
                            br, pack_off, deg);
                    }
                } else if (row_count == 2) {
                    GatherShiftDispatch<1, 2>(gath_h, lam_pad,
                        graph_meta_steady, row_shift_split,
                        br, pack_off, deg);
                } else {
                    GatherShiftDispatch<1, 1>(gath_h, lam_pad,
                        graph_meta_steady, row_shift_split,
                        br, pack_off, deg);
                }
                SetFlag<HardEvent::MTE3_V>(EVENT_ID7);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);

                Sub<half, false>(ext, ext, prev_h, (uint64_t)128, edge_reps, flat_bin_p);
                Abs<half, false>(new_msg_h, ext, (uint64_t)128, edge_reps, flat_abs_p);
                CompareScalar(row_mask, ext, (half)0.0f, CMPMODE::GE,
                              (uint64_t)128, edge_reps, sign_cmp_p);
                Select(sgn, row_mask, pos_one, neg_one,
                       SELMODE::VSEL_TENSOR_TENSOR_MODE,
                       (uint64_t)128, edge_reps, sign_sel_p);

                // T4/T6/T7: degree is one of nine BG1 constants.  Dispatch
                // once per row group, then run a fully unrolled reduction.
#define RUN_CHECK_DEG(D) \
                case D: CheckNodeDegree<D>(sign_prod, sgn, min1, min2, \
                                            new_msg_h, row_mask, row_lanes, \
                                            lane_reps, flat_bin_p, alpha_p, \
                                            lane_cmp_p, lane_sel_p, \
                                            lane_scale_p); break
                switch (deg) {
                    RUN_CHECK_DEG(3);
                    RUN_CHECK_DEG(4);
                    RUN_CHECK_DEG(5);
                    RUN_CHECK_DEG(6);
                    RUN_CHECK_DEG(7);
                    RUN_CHECK_DEG(8);
                    RUN_CHECK_DEG(9);
                    RUN_CHECK_DEG(10);
                    RUN_CHECK_DEG(19);
                }
#undef RUN_CHECK_DEG

                MulAddDst<half, half, false>(ext, sgn, new_msg_h,
                                             (uint64_t)128, edge_reps, flat_bin_p);
                // Keep the layered posterior in FP16 without per-edge
                // saturation.  The previous +/-20 Q8.8 clamp cost two full
                // Vector passes after every row; clamp once at final export.

                // Preserve messages only for the next iteration.  The 2D copy
                // converts edge-major UB back to contiguous per-CB GM rows.
                if (iter + 1 < MAX_ITER) {
                    MulCast<int8_t, half>(prev_i8, sgn, new_msg_h,
                                          (uint64_t)128, edge_reps, msg_cast_p);
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID6);
                    WaitFlag<HardEvent::V_MTE3>(EVENT_ID6);
                    store_prev.blockCount = (uint16_t)deg;
                    store_prev.srcStride =
                        (uint16_t)((row_lanes - 1) * PREV_EDGE_BLOCKS);
                    #pragma unroll 2
                    for (uint32_t rr = 0; rr < row_count; ++rr) {
                        #pragma unroll 2
                        for (uint32_t b = 0; b < batch; ++b) {
                            const size_t gm_off =
                                (size_t)(cb_base + b) * airan::PREV_ELEMS_PER_CB +
                                ub_off + rr * edge_elems;
                            DataCopy(prevMsgG[gm_off],
                                     prev_i8[(rr * batch + b) * Z], store_prev);
                        }
                    }
                    if (row_slot == 0) {
                        SetFlag<HardEvent::MTE3_V>(EVENT_ID4);
                    } else {
                        SetFlag<HardEvent::MTE3_V>(EVENT_ID5);
                    }
                    store_pending[row_slot] = true;
                }

                // T10: keep each posterior column in this edge's coordinate.
                // Fused rows have disjoint columns, so these writes cannot
                // alias.  The next gather consumes the precomputed delta.
                if (batch == 2) {
                    if (row_count == 2) {
                        ScatterResident<2, 2>(lam_pad, ext,
                            graph_meta_steady, pack_off, deg);
                    } else {
                        ScatterResident<2, 1>(lam_pad, ext,
                            graph_meta_steady, pack_off, deg);
                    }
                } else if (row_count == 2) {
                    ScatterResident<1, 2>(lam_pad, ext,
                        graph_meta_steady, pack_off, deg);
                } else {
                    ScatterResident<1, 1>(lam_pad, ext,
                        graph_meta_steady, pack_off, deg);
                }
                ub_off += row_count * edge_elems;
                pack_off += row_count * MAX_DEG;
                br += row_count;
            }
            SetFlag<HardEvent::MTE3_V>(EVENT_ID7);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
        }

        // Return from each column's last-edge coordinate to the public
        // variable-node order before exporting LLRs and making hard decisions.
        for (uint32_t bc = 0; bc < airan::LDPC_NFULL; ++bc) {
            const uint32_t coord = last_shift[bc];
            const uint32_t inv_anchor = (coord == 0) ? 0 : (Z - coord);
            RotateColumnInPlace(lam_pad[bc * batch * Z], gath_h,
                                inv_anchor, batch);
        }

        // One final clamp preserves the public Q8.8 posterior range while
        // avoiding two full edge-workspace passes after every check row.
        Mins(lam_pad, lam_pad, (half)(float)airan::LLR_CLIP_FX,
             (int32_t)lam_elems);
        Maxs(lam_pad, lam_pad, (half)-(float)airan::LLR_CLIP_FX,
             (int32_t)lam_elems);

        // Export posterior, then restore the half view for hard decisions.
        Cast(lam_pad_i16, lam_pad, RoundMode::CAST_RINT, (int32_t)lam_elems);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID5);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID5);
        if (batch == 1) {
            DataCopy(lamOutG[cb_lam_base], lam_pad_i16, lam_elems);
        } else {
            DataCopyParams store_lam;
            store_lam.blockCount = (uint16_t)batch;
            store_lam.blockLen = (uint16_t)(Z * sizeof(int16_t) / 32);
            store_lam.srcStride = 0;
            store_lam.dstStride = (uint16_t)((airan::LAM_ELEMS_PER_CB - Z) * sizeof(int16_t) / 32);
            for (uint32_t bc = 0; bc < airan::LDPC_NFULL; ++bc) {
                DataCopy(lamOutG[cb_lam_base + bc * Z],
                         lam_pad_i16[bc * batch * Z], store_lam);
            }
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID5);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID5);
        Cast(lam_pad, lam_pad_i16, RoundMode::CAST_NONE, (int32_t)lam_elems);

        constexpr uint32_t K = airan::LDPC_K;
        const uint32_t hard_elems = batch * K;
        auto hard_mask = bufPrevRow.Get<uint8_t>();
        auto ones_seg   = lam_pad[hard_elems];
        auto bits_h_seg = lam_pad[2 * hard_elems];
        Duplicate<half>(ones_seg, (half)1.0f, (int32_t)hard_elems);
        PipeBarrier<PIPE_V>();
        CompareScalar(hard_mask, lam_pad, (half)0.0f, CMPMODE::LT, hard_elems);
        PipeBarrier<PIPE_V>();
        Select(bits_h_seg, hard_mask, ones_seg, (half)0.0f,
               SELMODE::VSEL_TENSOR_SCALAR_MODE, hard_elems);
        PipeBarrier<PIPE_V>();
        Cast(bits_ub, bits_h_seg, RoundMode::CAST_NONE, (int32_t)hard_elems);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID7);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID7);
        if (batch == 1) {
            DataCopy(bitsOutG[(size_t)cb_base * K], bits_ub, K);
        } else {
            DataCopyParams store_bits;
            store_bits.blockCount = (uint16_t)(K / Z);
            store_bits.blockLen = (uint16_t)(Z * sizeof(int8_t) / 32);
            store_bits.srcStride = (uint16_t)((batch - 1) * Z * sizeof(int8_t) / 32);
            store_bits.dstStride = 0;
            for (uint32_t b = 0; b < batch; ++b) {
                DataCopy(bitsOutG[(size_t)(cb_base + b) * K], bits_ub[b * Z], store_bits);
            }
        }
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}
