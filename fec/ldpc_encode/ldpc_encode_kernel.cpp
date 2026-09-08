









#include "kernel_operator.h"
#include "ldpc_encode.h"

using namespace AscendC;

namespace {

constexpr uint32_t UB_INFO_PAD_ELEMS = airan::LDPC_KB * airan::Z_PAD;
constexpr uint32_t UB_AX_PAD_ELEMS   = airan::LDPC_G  * airan::Z_PAD;
constexpr uint32_t UB_PA_PAD_ELEMS   = airan::LDPC_G  * airan::Z_PAD;

constexpr uint32_t UB_INFO_PAD_BYTES = UB_INFO_PAD_ELEMS * sizeof(int16_t);
constexpr uint32_t UB_AX_PAD_BYTES   = UB_AX_PAD_ELEMS   * sizeof(int16_t);
constexpr uint32_t UB_PA_PAD_BYTES   = UB_PA_PAD_ELEMS   * sizeof(int16_t);
constexpr uint32_t UB_PB_BYTES       = airan::LDPC_MB4Z  * sizeof(int16_t);

constexpr uint32_t UB_INFO_I8_BYTES  = airan::LDPC_K;
constexpr uint32_t UB_HALF_BYTES     = airan::LDPC_MB4Z * sizeof(half);
constexpr uint32_t UB_OUT_I8_BYTES   = airan::LDPC_N_RAW;

constexpr uint32_t ALIGN16 = 16;
constexpr uint32_t SHIFT_A_PADDED  = ((airan::SHIFT_A_ELEMS  + ALIGN16 - 1) / ALIGN16) * ALIGN16;
constexpr uint32_t SHIFT_BI_PADDED = ((airan::SHIFT_BI_ELEMS + ALIGN16 - 1) / ALIGN16) * ALIGN16;
constexpr uint32_t SHIFT_C_PADDED  = ((airan::SHIFT_C_ELEMS  + ALIGN16 - 1) / ALIGN16) * ALIGN16;
constexpr uint32_t SHIFT_D_PADDED  = ((airan::SHIFT_D_ELEMS  + ALIGN16 - 1) / ALIGN16) * ALIGN16;

constexpr uint32_t UB_SHIFT_A_BYTES  = SHIFT_A_PADDED  * sizeof(int16_t);
constexpr uint32_t UB_SHIFT_BI_BYTES = SHIFT_BI_PADDED * sizeof(int16_t);
constexpr uint32_t UB_SHIFT_C_BYTES  = SHIFT_C_PADDED  * sizeof(int16_t);
constexpr uint32_t UB_SHIFT_D_BYTES  = SHIFT_D_PADDED  * sizeof(int16_t);

constexpr uint32_t UB_ONES_BYTES     = airan::LDPC_Z * sizeof(int16_t);

}





__aicore__ inline void CompactShiftRows(
    LocalTensor<int16_t>& table,
    uint32_t              num_rows,
    uint32_t              num_block_cols,
    uint32_t              slots_per_col)
{
    const uint32_t row_width = num_block_cols * slots_per_col;
    for (uint32_t br = 0; br < num_rows; ++br) {
        const uint32_t row_base = br * row_width;
        uint32_t write = 0;
        for (uint32_t bc = 0; bc < num_block_cols; ++bc) {
            for (uint32_t slot = 0; slot < slots_per_col; ++slot) {
                const int16_t shift = table.GetValue(row_base + bc * slots_per_col + slot);
                if (shift >= 0) {
                    table.SetValue(row_base + write,
                                   (int16_t)(bc * airan::Z_PAD + (uint32_t)shift));
                    ++write;
                }
            }
        }
        for (uint32_t i = write; i < row_width; ++i) {
            table.SetValue(row_base + i, (int16_t)-1);
        }
    }
}

__aicore__ inline void AccumulatePackedEdges(
    LocalTensor<int16_t>& dst,
    LocalTensor<int16_t>& src_base,
    LocalTensor<int16_t>& packed_row,
    uint32_t              row_width,
    uint32_t              first_edge)
{
    for (uint32_t i = first_edge; i < row_width; ++i) {
        const int32_t packed = (int32_t)packed_row.GetValue(i);
        if (packed < 0) break;
        Add(dst, dst, src_base[(uint32_t)packed], (int32_t)airan::LDPC_Z);
    }
}

__aicore__ inline void SpmvBlockRow(
    LocalTensor<int16_t>& dst_block,
    LocalTensor<int16_t>& src_base,
    LocalTensor<int16_t>& packed_row,
    uint32_t              row_width,
    LocalTensor<int16_t>& ones_ub)
{
    const int32_t first = (int32_t)packed_row.GetValue(0);
    if (first < 0) {
        Duplicate<int16_t>(dst_block, (int16_t)0, airan::LDPC_Z);
        return;
    }
    Adds(dst_block, src_base[(uint32_t)first], (int16_t)0, (int32_t)airan::LDPC_Z);
    AccumulatePackedEdges(dst_block, src_base, packed_row, row_width, 1);
    And(dst_block, dst_block, ones_ub, (int32_t)airan::LDPC_Z);
}

__aicore__ inline void SpmvBlockRow_CD(
    LocalTensor<int16_t>& dst_block,
    uint32_t              dst_offset_elems,
    LocalTensor<int16_t>& info_base,
    LocalTensor<int16_t>& pa_base,
    LocalTensor<int16_t>& packed_c_row,
    LocalTensor<int16_t>& packed_d_row,
    LocalTensor<int16_t>& ones_ub)
{
    auto dst = dst_block[dst_offset_elems];
    const int32_t first_c = (int32_t)packed_c_row.GetValue(0);
    const int32_t first_d = (int32_t)packed_d_row.GetValue(0);
    uint32_t c_begin = 0;
    uint32_t d_begin = 0;

    if (first_c >= 0) {
        Adds(dst, info_base[(uint32_t)first_c], (int16_t)0, (int32_t)airan::LDPC_Z);
        c_begin = 1;
    } else if (first_d >= 0) {
        Adds(dst, pa_base[(uint32_t)first_d], (int16_t)0, (int32_t)airan::LDPC_Z);
        d_begin = 1;
    } else {
        Duplicate<int16_t>(dst, (int16_t)0, airan::LDPC_Z);
        return;
    }

    AccumulatePackedEdges(dst, info_base, packed_c_row, airan::LDPC_KB, c_begin);
    AccumulatePackedEdges(dst, pa_base, packed_d_row, airan::LDPC_G, d_begin);
    And(dst, dst, ones_ub, (int32_t)airan::LDPC_Z);
}


extern "C" __global__ __aicore__ void ldpc_encode_kernel(
    GM_ADDR info_gm,
    GM_ADDR shift_a_gm,
    GM_ADDR shift_bi_gm,
    GM_ADDR shift_c_gm,
    GM_ADDR shift_d_gm,
    GM_ADDR dbg_gm,
    GM_ADDR output_gm)
{

    const uint32_t raw_bid = GetBlockIdx();
    const uint32_t aiv_id  = raw_bid ^ 2;









    if (aiv_id >= airan::LDPC_C_NUM) return;

    GlobalTensor<int8_t>  infoG;
    GlobalTensor<int16_t> outputG_i16;
    GlobalTensor<int16_t> shiftAG, shiftBiG, shiftCG, shiftDG;
    GlobalTensor<int32_t> dbgG;

    infoG.SetGlobalBuffer       ((__gm__ int8_t*)  info_gm,    airan::INFO_BYTES);
    shiftAG.SetGlobalBuffer     ((__gm__ int16_t*) shift_a_gm, SHIFT_A_PADDED);
    shiftBiG.SetGlobalBuffer    ((__gm__ int16_t*) shift_bi_gm,SHIFT_BI_PADDED);
    shiftCG.SetGlobalBuffer     ((__gm__ int16_t*) shift_c_gm, SHIFT_C_PADDED);
    shiftDG.SetGlobalBuffer     ((__gm__ int16_t*) shift_d_gm, SHIFT_D_PADDED);
    outputG_i16.SetGlobalBuffer ((__gm__ int16_t*) output_gm,  (airan::LDPC_C_NUM * airan::LDPC_N_RAW) / 2);
    dbgG.SetGlobalBuffer        ((__gm__ int32_t*) dbg_gm,     airan::LDPC_C_NUM * airan::DBG_WORDS_PER_CB);

    TPipe pipe;
    TBuf<TPosition::VECCALC> bufInfoPad, bufAxPad, bufPaPad, bufPb;
    TBuf<TPosition::VECCALC> bufInfoI8, bufHalf, bufOutI8;
    TBuf<TPosition::VECCALC> bufShiftA, bufShiftBi, bufShiftC, bufShiftD;
    TBuf<TPosition::VECCALC> bufOnes;

    pipe.InitBuffer(bufInfoPad, UB_INFO_PAD_BYTES);
    pipe.InitBuffer(bufAxPad,   UB_AX_PAD_BYTES);
    pipe.InitBuffer(bufPaPad,   UB_PA_PAD_BYTES);
    pipe.InitBuffer(bufPb,      UB_PB_BYTES);
    pipe.InitBuffer(bufInfoI8,  UB_INFO_I8_BYTES);
    pipe.InitBuffer(bufHalf,    UB_HALF_BYTES);
    pipe.InitBuffer(bufOutI8,   UB_OUT_I8_BYTES);
    pipe.InitBuffer(bufShiftA,  UB_SHIFT_A_BYTES);
    pipe.InitBuffer(bufShiftBi, UB_SHIFT_BI_BYTES);
    pipe.InitBuffer(bufShiftC,  UB_SHIFT_C_BYTES);
    pipe.InitBuffer(bufShiftD,  UB_SHIFT_D_BYTES);
    pipe.InitBuffer(bufOnes,    UB_ONES_BYTES);

    auto info_pad  = bufInfoPad.Get<int16_t>();
    auto ax_pad    = bufAxPad.Get<int16_t>();
    auto pa_pad    = bufPaPad.Get<int16_t>();
    auto pb_ub     = bufPb.Get<int16_t>();
    auto info_i8   = bufInfoI8.Get<int8_t>();
    auto half_ub   = bufHalf.Get<half>();
    auto out_i8_ub = bufOutI8.Get<int8_t>();
    auto shift_a   = bufShiftA.Get<int16_t>();
    auto shift_bi  = bufShiftBi.Get<int16_t>();
    auto shift_c   = bufShiftC.Get<int16_t>();
    auto shift_d   = bufShiftD.Get<int16_t>();
    auto ones_ub   = bufOnes.Get<int16_t>();




    Duplicate<int16_t>(ones_ub, (int16_t)1, airan::LDPC_Z);

    DataCopy(shift_a,  shiftAG,  SHIFT_A_PADDED);
    DataCopy(shift_bi, shiftBiG, SHIFT_BI_PADDED);
    DataCopy(shift_c,  shiftCG,  SHIFT_C_PADDED);
    DataCopy(shift_d,  shiftDG,  SHIFT_D_PADDED);
    PipeBarrier<PIPE_ALL>();

    CompactShiftRows(shift_a,  airan::LDPC_G,    airan::LDPC_KB, 1);
    CompactShiftRows(shift_bi, airan::LDPC_G,    airan::LDPC_G,  airan::SHIFT_BI_MAXW);
    CompactShiftRows(shift_c,  airan::LDPC_MBMG, airan::LDPC_KB, 1);
    CompactShiftRows(shift_d,  airan::LDPC_MBMG, airan::LDPC_G,  1);
    PipeBarrier<PIPE_ALL>();




    for (uint32_t cb = aiv_id; cb < airan::LDPC_C_NUM; cb += 4) {


        DataCopy(info_i8, infoG[cb * airan::LDPC_K], airan::LDPC_K);
        PipeBarrier<PIPE_ALL>();

        Cast(half_ub, info_i8, RoundMode::CAST_NONE, airan::LDPC_K);
        PipeBarrier<PIPE_V>();

        for (uint32_t bc = 0; bc < airan::LDPC_KB; ++bc) {
            Cast(info_pad[bc * airan::Z_PAD],
                 half_ub[bc * airan::LDPC_Z],
                 RoundMode::CAST_RINT,
                 airan::LDPC_Z);
        }
        PipeBarrier<PIPE_V>();

        for (uint32_t bc = 0; bc < airan::LDPC_KB; ++bc) {
            uint32_t dst = bc * airan::Z_PAD;
            Adds(info_pad[dst + airan::LDPC_Z], info_pad[dst],
                 (int16_t)0, (int32_t)airan::LDPC_Z);
        }
        PipeBarrier<PIPE_V>();


        for (uint32_t br = 0; br < airan::LDPC_G; ++br) {
            auto shift_row = shift_a[br * airan::LDPC_KB];
            auto dst_block = ax_pad[br * airan::Z_PAD];
            SpmvBlockRow(dst_block, info_pad, shift_row, airan::LDPC_KB, ones_ub);
        }
        for (uint32_t br = 0; br < airan::LDPC_G; ++br) {
            uint32_t dst = br * airan::Z_PAD;
            Adds(ax_pad[dst + airan::LDPC_Z], ax_pad[dst], (int16_t)0, (int32_t)airan::LDPC_Z);
        }
        PipeBarrier<PIPE_V>();


        for (uint32_t br = 0; br < airan::LDPC_G; ++br) {
            auto shift_bi_row = shift_bi[br * airan::LDPC_G * airan::SHIFT_BI_MAXW];
            auto dst_block    = pa_pad[br * airan::Z_PAD];
            SpmvBlockRow(dst_block, ax_pad, shift_bi_row,
                         airan::LDPC_G * airan::SHIFT_BI_MAXW, ones_ub);
        }
        for (uint32_t br = 0; br < airan::LDPC_G; ++br) {
            uint32_t dst = br * airan::Z_PAD;
            Adds(pa_pad[dst + airan::LDPC_Z], pa_pad[dst], (int16_t)0, (int32_t)airan::LDPC_Z);
        }
        PipeBarrier<PIPE_V>();


        for (uint32_t br = 0; br < airan::LDPC_MBMG; ++br) {
            auto shift_c_row = shift_c[br * airan::LDPC_KB];
            auto shift_d_row = shift_d[br * airan::LDPC_G];
            SpmvBlockRow_CD(pb_ub, br * airan::LDPC_Z,
                            info_pad, pa_pad,
                            shift_c_row, shift_d_row,
                            ones_ub);
        }
        PipeBarrier<PIPE_V>();


        {
            auto out_i16  = out_i8_ub.ReinterpretCast<int16_t>();
            auto info_i16 = info_i8.ReinterpretCast<int16_t>();
            Adds(out_i16, info_i16[airan::LDPC_Z], (int16_t)0,
                 (int32_t)(airan::LDPC_INFO_OUT / 2));
        }
        PipeBarrier<PIPE_V>();


        for (uint32_t bc = 0; bc < airan::LDPC_G; ++bc) {
            Cast(half_ub, pa_pad[bc * airan::Z_PAD], RoundMode::CAST_NONE, airan::LDPC_Z);
            PipeBarrier<PIPE_V>();
            Cast(out_i8_ub[airan::LDPC_INFO_OUT + bc * airan::LDPC_Z],
                 half_ub, RoundMode::CAST_NONE, airan::LDPC_Z);
            PipeBarrier<PIPE_V>();
        }


        Cast(half_ub, pb_ub, RoundMode::CAST_NONE, airan::LDPC_MB4Z);
        PipeBarrier<PIPE_V>();
        Cast(out_i8_ub[airan::LDPC_INFO_OUT + airan::LDPC_FZ], half_ub,
             RoundMode::CAST_NONE, airan::LDPC_MB4Z);
        PipeBarrier<PIPE_ALL>();


        auto out_i16_view = out_i8_ub.ReinterpretCast<int16_t>();
        DataCopy(outputG_i16[cb * (airan::LDPC_N_RAW / 2)],
                 out_i16_view, airan::LDPC_N_RAW / 2);
        PipeBarrier<PIPE_ALL>();
    }
}
