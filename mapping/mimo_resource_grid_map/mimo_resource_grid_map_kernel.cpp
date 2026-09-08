








#include "kernel_operator.h"
#include "mimo_resource_grid_map.h"

using namespace AscendC;

namespace {
namespace rgm = airan::mimo_resource_grid_map;
constexpr uint32_t ZERO_SOURCE = rgm::N_DMRS_PAD;
constexpr uint32_t DMRS_SOURCE_PAD = rgm::N_DMRS_PAD + 16;
}

class MimoResourceGridMap {
public:
    __aicore__ inline MimoResourceGridMap() {}

    __aicore__ inline void Init(GM_ADDR layer_re_gm, GM_ADDR layer_im_gm,
                                GM_ADDR dmrs_re_gm, GM_ADDR dmrs_im_gm,
                                GM_ADDR data_dst_offset_gm, GM_ADDR dmrs_dst_offset_gm,
                                GM_ADDR grid_re_gm, GM_ADDR grid_im_gm,
                                GM_ADDR tiling_gm, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void MapDataPlane(const GlobalTensor<half> &source,
                                        const GlobalTensor<half> &destination,
                                        bool useSecondBuffer);
    __aicore__ inline void EmitDmrsPlane(const GlobalTensor<half> &source,
                                         const GlobalTensor<half> &destination,
                                         uint32_t sourceBase, uint32_t symbol);
    __aicore__ inline void MapDmrs();

    TPipe *pipe_;
    uint32_t layer_;
    uint32_t numLayers_;
    uint32_t numDmrs_;
    uint32_t dataStride_;
    uint32_t gridStride_;
    uint32_t dmrsSymbolMask_;
    uint32_t dmrsSymbol0_;
    uint32_t dmrsSymbol1_;

    GlobalTensor<half> layerReG_, layerImG_, dmrsReG_, dmrsImG_;
    GlobalTensor<half> gridReG_, gridImG_;
    GlobalTensor<uint32_t> dataOffsetG_, dmrsOffsetG_, metadataG_;

    TBuf<TPosition::VECCALC> bufDataGroup_;
    TBuf<TPosition::VECCALC> bufDataIm_;
    TBuf<TPosition::VECCALC> bufDmrs_;
    TBuf<TPosition::VECCALC> bufRow_;
    TBuf<TPosition::VECCALC> bufDmrsOffset_;
    TBuf<TPosition::VECCALC> bufInverse_;
    TBuf<TPosition::VECCALC> bufMetadata_;
};

__aicore__ inline void MimoResourceGridMap::Init(
    GM_ADDR layer_re_gm, GM_ADDR layer_im_gm,
    GM_ADDR dmrs_re_gm, GM_ADDR dmrs_im_gm,
    GM_ADDR data_dst_offset_gm, GM_ADDR dmrs_dst_offset_gm,
    GM_ADDR grid_re_gm, GM_ADDR grid_im_gm,
    GM_ADDR tiling_gm, TPipe *pipe)
{
    pipe_ = pipe;
    layer_ = GetBlockIdx();
    numLayers_ = 0;
    numDmrs_ = 0;
    dataStride_ = 0;
    gridStride_ = 0;
    dmrsSymbol0_ = 0;
    dmrsSymbol1_ = 0;

    layerReG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(layer_re_gm),
                              rgm::MAX_LAYERS * rgm::N_DATA_PAD);
    layerImG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(layer_im_gm),
                              rgm::MAX_LAYERS * rgm::N_DATA_PAD);
    dmrsReG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(dmrs_re_gm),
                             rgm::MAX_LAYERS * rgm::CURRENT_DMRS_SYMBOLS * rgm::N_DMRS_PAD);
    dmrsImG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(dmrs_im_gm),
                             rgm::MAX_LAYERS * rgm::CURRENT_DMRS_SYMBOLS * rgm::N_DMRS_PAD);
    dataOffsetG_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(data_dst_offset_gm),
                                 rgm::N_DATA_PAD);
    dmrsOffsetG_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(dmrs_dst_offset_gm),
                                 rgm::MAX_LAYERS * rgm::CURRENT_DMRS_SYMBOLS * rgm::N_DMRS_PAD);
    gridReG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(grid_re_gm),
                             rgm::MAX_LAYERS * rgm::N_GRID);
    gridImG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(grid_im_gm),
                             rgm::MAX_LAYERS * rgm::N_GRID);
    metadataG_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tiling_gm), rgm::META_WORDS);

    pipe_->InitBuffer(bufDataGroup_, rgm::N_DATA_PAD * sizeof(half));
    pipe_->InitBuffer(bufDataIm_, rgm::N_DATA_PAD * sizeof(half));
    pipe_->InitBuffer(bufDmrs_, DMRS_SOURCE_PAD * sizeof(half));
    pipe_->InitBuffer(bufRow_, rgm::N_SC_PAD * sizeof(half));
    pipe_->InitBuffer(bufDmrsOffset_, rgm::N_DMRS_PAD * sizeof(uint32_t));
    pipe_->InitBuffer(bufInverse_, rgm::N_SC_PAD * sizeof(uint32_t));
    pipe_->InitBuffer(bufMetadata_, rgm::TILING_BYTES);
}

__aicore__ inline void MimoResourceGridMap::MapDataPlane(
    const GlobalTensor<half> &source, const GlobalTensor<half> &destination,
    bool useSecondBuffer)
{
    LocalTensor<half> compact = useSecondBuffer ? bufDataIm_.Get<half>() : bufDataGroup_.Get<half>();
    auto row = bufRow_.Get<half>();
    const half zero = static_cast<half>(0.0f);
    const uint32_t sourceLayer = layer_ * dataStride_;
    const uint32_t destinationLayer = layer_ * gridStride_;
    constexpr uint32_t groups = rgm::N_DATA_RE / rgm::DATA_GROUP_RE;



    for (uint32_t group = 0; group < groups; ++group) {
        const uint32_t groupSource = group * rgm::DATA_GROUP_RE;
        DataCopy(compact[groupSource], source[sourceLayer + groupSource], rgm::DATA_GROUP_RE);
    }
    auto eM2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);

    for (uint32_t ordinal = 0; ordinal < rgm::N_DATA_RE / rgm::N_SC_USED; ++ordinal) {
        const uint32_t compactOffset = ordinal * rgm::N_SC_USED;
        uint32_t symbol = ordinal;
        if (symbol >= dmrsSymbol0_) ++symbol;
        if (symbol >= dmrsSymbol1_) ++symbol;

        Adds(row, compact[compactOffset], zero, rgm::N_SC_USED);
        Duplicate(row[rgm::N_SC_USED], zero, rgm::N_SC_PAD - rgm::N_SC_USED);
        PipeBarrier<PIPE_V>();
        auto eV3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eV3); WaitFlag<HardEvent::V_MTE3>(eV3);
        DataCopy(destination[destinationLayer + symbol * rgm::N_SC_PAD], row, rgm::N_SC_PAD);
        auto e3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(e3V); WaitFlag<HardEvent::MTE3_V>(e3V);
    }
    auto eVM2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eVM2); WaitFlag<HardEvent::V_MTE2>(eVM2);
}

__aicore__ inline void MimoResourceGridMap::EmitDmrsPlane(
    const GlobalTensor<half> &source, const GlobalTensor<half> &destination,
    uint32_t sourceBase, uint32_t symbol)
{
    auto dmrs = bufDmrs_.Get<half>();
    auto row = bufRow_.Get<half>();
    auto inverse = bufInverse_.Get<uint32_t>();
    const half zero = static_cast<half>(0.0f);
    const uint32_t destinationLayer = layer_ * gridStride_;

    DataCopy(dmrs, source[sourceBase], rgm::N_DMRS_PAD);
    auto eM2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eM2V); WaitFlag<HardEvent::MTE2_V>(eM2V);
    Duplicate(dmrs[ZERO_SOURCE], zero, DMRS_SOURCE_PAD - ZERO_SOURCE);
    PipeBarrier<PIPE_V>();
    Gather(row, dmrs, inverse, static_cast<uint32_t>(0), rgm::N_SC_PAD);
    PipeBarrier<PIPE_V>();
    auto eV3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eV3); WaitFlag<HardEvent::V_MTE3>(eV3);
    DataCopy(destination[destinationLayer + symbol * rgm::N_SC_PAD], row, rgm::N_SC_PAD);
    auto e3M2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(e3M2); WaitFlag<HardEvent::MTE3_MTE2>(e3M2);
}

__aicore__ inline void MimoResourceGridMap::MapDmrs()
{
    auto offset = bufDmrsOffset_.Get<uint32_t>();
    auto inverse = bufInverse_.Get<uint32_t>();




    const uint32_t firstSourceBase = layer_ * numDmrs_ * rgm::N_DMRS_PAD;
    DataCopy(offset, dmrsOffsetG_[firstSourceBase], rgm::N_DMRS_PAD);
    auto eM2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eM2S); WaitFlag<HardEvent::MTE2_S>(eM2S);
    Duplicate(inverse, static_cast<uint32_t>(ZERO_SOURCE * sizeof(half)), rgm::N_SC_PAD);
    auto eVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eVS); WaitFlag<HardEvent::V_S>(eVS);

    uint32_t firstSymbol = 0;
    for (uint32_t pilot = 0; pilot < rgm::N_DMRS_RE; ++pilot) {
        const uint32_t byteOffset = offset.GetValue(pilot);
        const uint32_t elementOffset = byteOffset / sizeof(half);
        firstSymbol = elementOffset / rgm::N_SC_PAD;
        const uint32_t subcarrier = elementOffset - firstSymbol * rgm::N_SC_PAD;
        inverse.SetValue(subcarrier, pilot * sizeof(half));
    }
    auto eSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eSV); WaitFlag<HardEvent::S_V>(eSV);

    for (uint32_t dmrsIndex = 0; dmrsIndex < numDmrs_; ++dmrsIndex) {
        const uint32_t sourceBase = (layer_ * numDmrs_ + dmrsIndex) * rgm::N_DMRS_PAD;
        uint32_t symbol = firstSymbol;
        if (dmrsIndex != 0) {

            DataCopy(offset, dmrsOffsetG_[sourceBase], 16);
            auto eNextM2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(eNextM2S); WaitFlag<HardEvent::MTE2_S>(eNextM2S);
            symbol = (offset.GetValue(0) / sizeof(half)) / rgm::N_SC_PAD;
        }


        EmitDmrsPlane(dmrsReG_, gridReG_, sourceBase, symbol);
        EmitDmrsPlane(dmrsImG_, gridImG_, sourceBase, symbol);
    }
}

__aicore__ inline void MimoResourceGridMap::Process()
{
    auto metadata = bufMetadata_.Get<uint32_t>();
    DataCopy(metadata, metadataG_, rgm::META_WORDS);
    auto eM2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eM2S); WaitFlag<HardEvent::MTE2_S>(eM2S);

    if (metadata.GetValue(0) != rgm::META_MAGIC) return;
    numLayers_ = metadata.GetValue(1);
    numDmrs_ = metadata.GetValue(2);
    dataStride_ = metadata.GetValue(7);
    gridStride_ = metadata.GetValue(9);
    dmrsSymbolMask_ = metadata.GetValue(11);
    uint32_t dmrsFound = 0;
    for (uint32_t symbol = 0; symbol < rgm::N_SYMBOLS; ++symbol) {
        if (((dmrsSymbolMask_ >> symbol) & 1u) == 0) continue;
        if (dmrsFound == 0) dmrsSymbol0_ = symbol;
        else dmrsSymbol1_ = symbol;
        ++dmrsFound;
    }
    if (layer_ >= numLayers_ || numLayers_ > rgm::MAX_LAYERS ||
        numDmrs_ != rgm::CURRENT_DMRS_SYMBOLS || dataStride_ != rgm::N_DATA_PAD ||
        gridStride_ != rgm::N_GRID) {
        return;
    }

    MapDataPlane(layerReG_, gridReG_, false);
    MapDataPlane(layerImG_, gridImG_, true);
    MapDmrs();
}

extern "C" __global__ __aicore__ void mimo_resource_grid_map_kernel(
    GM_ADDR layer_re_gm, GM_ADDR layer_im_gm,
    GM_ADDR dmrs_re_gm, GM_ADDR dmrs_im_gm,
    GM_ADDR data_dst_offset_gm, GM_ADDR dmrs_dst_offset_gm,
    GM_ADDR grid_re_gm, GM_ADDR grid_im_gm,
    GM_ADDR workspace_gm, GM_ADDR tiling_gm)
{
    (void)workspace_gm;
    TPipe pipe;
    MimoResourceGridMap op;
    op.Init(layer_re_gm, layer_im_gm, dmrs_re_gm, dmrs_im_gm,
            data_dst_offset_gm, dmrs_dst_offset_gm, grid_re_gm, grid_im_gm,
            tiling_gm, &pipe);
    op.Process();
}
