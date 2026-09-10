#include <acl/acl.h>

#include "aclrtlaunch_channel_est_lmmse_kernel.h"
#include "aclrtlaunch_channel_est_lmmse_pack_natural_kernel.h"
#include "aclrtlaunch_channel_est_lmmse_pad_layers_kernel.h"
#include "aclrtlaunch_ldpc_decode_kernel.h"
#include "aclrtlaunch_layer_demap_kernel.h"
#include "aclrtlaunch_mimo_detect_bri_kernel.h"
#include "aclrtlaunch_mimo_dmrs_ls_kernel.h"
#include "aclrtlaunch_ofdm_demod_batch_kernel.h"

#include "bri_grouped_rhs_adapter.h"
#include "channel_est_lmmse.h"
#include "descramble_mimo.h"
#define TILING_TOTAL_SIZE LDPC_TILING_TOTAL_SIZE
#include "ldpc_decode.h"
#undef TILING_TOTAL_SIZE
#include "layer_demap.h"
#include "mimo_detect_bri.h"
#include "mimo_detect_io_pack.h"
#include "mimo_dmrs_gen.h"
#include "mimo_dmrs_ls.h"
#include "ofdm_demod.h"
#include "pusch_mimo_runtime_config.h"
#include "qam256_demod_batch.h"
#include "rate_dematch_mimo.h"
#include "re_demap_batch.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void GenerateOfdmDemodTiling(const char *, uint8_t *);
extern "C" void GenerateChannelEstLmmseTiling(const char *, uint8_t *);
extern "C" void GenerateMimoDetectBriTiling(const char *, uint8_t *);

namespace fs = std::filesystem;
namespace ofdm = ofdm_demod_batch;
namespace rd = airan::re_demap_batch;
namespace dg = airan::mimo_dmrs_gen;
namespace ls = airan::mimo_dmrs_ls;
namespace ce = airan::channel_est_lmmse;
namespace io = airan::mimo_detect_io_pack;
namespace qam = airan::qam256_demod_batch;
namespace layer = airan::layer_demap;
namespace descr = airan::descramble_mimo;
namespace rate = airan::rate_dematch_mimo;

namespace {

constexpr uint16_t kRank = 4;
constexpr uint16_t kRx = 64;
constexpr uint16_t kLayerCapacity = 16;
constexpr uint16_t kSlots = 23;
constexpr size_t kHalfBytes = sizeof(uint16_t);
constexpr size_t kSharedWorkspaceBytes = 16u * 1024u * 1024u;
constexpr size_t kOfdmTilingBytes = 2u * sizeof(optiling::TCubeTiling);
constexpr size_t kGroupedYElems =
    static_cast<size_t>(io::N_RE / 8u) * io::NR * io::NL;

void CheckAcl(aclError status, const char *expression)
{
    if (status != ACL_ERROR_NONE) {
        throw std::runtime_error(std::string(expression) + " ACL status=" +
                                 std::to_string(status));
    }
}

#define ACL_CHECK(expression) CheckAcl((expression), #expression)

void CheckLaunch(uint32_t status, const char *name)
{
    if (status != ACL_ERROR_NONE) {
        throw std::runtime_error(std::string(name) + " launch status=" +
                                 std::to_string(status));
    }
}

template <typename T>
std::vector<T> ReadExact(const fs::path &path, size_t count)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const size_t bytes = count * sizeof(T);
    if (!input || static_cast<size_t>(input.tellg()) != bytes) {
        throw std::runtime_error(path.string() + " expected exactly " +
                                 std::to_string(bytes) + " bytes");
    }
    std::vector<T> result(count);
    input.seekg(0);
    input.read(reinterpret_cast<char *>(result.data()),
               static_cast<std::streamsize>(bytes));
    if (!input) throw std::runtime_error("short read: " + path.string());
    return result;
}

template <typename T>
void WriteExact(const fs::path &path, const std::vector<T> &value)
{
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char *>(value.data()),
                 static_cast<std::streamsize>(value.size() * sizeof(T)));
    if (!output) throw std::runtime_error("write failed: " + path.string());
}

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t bytes) { Allocate(bytes); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    ~DeviceBuffer() { if (data_ != nullptr) aclrtFree(data_); }

    void Allocate(size_t bytes)
    {
        if (data_ != nullptr || bytes == 0) throw std::runtime_error("bad device allocation");
        ACL_CHECK(aclrtMalloc(&data_, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        bytes_ = bytes;
    }
    void Upload(const void *source, size_t bytes)
    {
        if (bytes > bytes_) throw std::runtime_error("device upload overflow");
        ACL_CHECK(aclrtMemcpy(data_, bytes_, source, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    template <typename T> void Upload(const std::vector<T> &source)
    {
        Upload(source.data(), source.size() * sizeof(T));
    }
    void *get() const { return data_; }
    size_t bytes() const { return bytes_; }
    void *offset(size_t bytes) const
    {
        if (bytes > bytes_) throw std::runtime_error("device offset overflow");
        return static_cast<uint8_t *>(data_) + bytes;
    }
private:
    void *data_ = nullptr;
    size_t bytes_ = 0;
};

struct EventTimer {
    EventTimer()
    {
        ACL_CHECK(aclrtCreateEvent(&start));
        ACL_CHECK(aclrtCreateEvent(&end));
    }
    ~EventTimer()
    {
        if (start != nullptr) aclrtDestroyEvent(start);
        if (end != nullptr) aclrtDestroyEvent(end);
    }
    void Begin(aclrtStream stream) { ACL_CHECK(aclrtRecordEvent(start, stream)); }
    void End(aclrtStream stream) { ACL_CHECK(aclrtRecordEvent(end, stream)); }
    double ReadUs()
    {
        float ms = 0.0f;
        ACL_CHECK(aclrtEventElapsedTime(&ms, start, end));
        return static_cast<double>(ms) * 1000.0;
    }
    aclrtEvent start = nullptr;
    aclrtEvent end = nullptr;
};

struct Options {
    fs::path input_root;
    fs::path output;
    fs::path receipt;
    uint16_t rank = kRank;
    uint16_t slots = kSlots;
};

Options ParseOptions(int argc, char **argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        auto value = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value for " + argument);
            return argv[i];
        };
        if (argument == "--input-root") options.input_root = value();
        else if (argument == "--output") options.output = value();
        else if (argument == "--receipt") options.receipt = value();
        else if (argument == "--rank") options.rank = static_cast<uint16_t>(std::stoul(value()));
        else if (argument == "--slots") options.slots = static_cast<uint16_t>(std::stoul(value()));
        else throw std::runtime_error("unknown option: " + argument);
    }
    if (options.input_root.empty() || options.output.empty() || options.receipt.empty() ||
        options.rank != kRank || options.slots != kSlots) {
        throw std::runtime_error(
            "this fused image requires --input-root DIR --output FILE --receipt FILE "
            "--rank 4 --slots 23");
    }
    return options;
}

double Median(std::vector<double> value)
{
    if (value.empty()) return 0.0;
    std::sort(value.begin(), value.end());
    return value[value.size() / 2];
}

void AppendMetric(std::map<std::string, std::vector<double>> *metrics,
                  const std::string &name, EventTimer *timer)
{
    (*metrics)[name].push_back(timer->ReadUs());
}

airan::PuschMimoConfig LoadConfig(uint16_t slot,
                                  airan::PuschMimoRuntimeConfig *runtime)
{
    bool enabled = false;
    std::string why;
    if (airan::LoadMimoRuntimeConfigFromEnv(kRank, runtime, &enabled, &why) !=
            airan::MimoRuntimeConfigStatus::kSuccess || !enabled) {
        throw std::runtime_error("runtime profile is required: " + why);
    }
    if (runtime->num_tx_antennas != 16 || runtime->num_rx_antennas != kRx ||
        runtime->rx_bucket != kRx || runtime->layer_bucket != kLayerCapacity ||
        runtime->num_layers != kRank || runtime->num_symbols != 14 ||
        runtime->used_subcarriers != 1596 || runtime->padded_subcarriers != 1664) {
        throw std::runtime_error("runtime profile is not fd16x64 Rank4");
    }
    airan::PuschMimoConfig config{};
    if (airan::DerivePuschMimoConfig(*runtime, kRx, &config, &why, slot) !=
        airan::MimoRuntimeConfigStatus::kSuccess) {
        throw std::runtime_error("cannot derive runtime config: " + why);
    }
    return config;
}

struct StaticWeights {
    DeviceBuffer ofdm_w32r, ofdm_w32i, ofdm_w64r, ofdm_w64i, ofdm_twr, ofdm_twi;
    DeviceBuffer ce_br, ce_bi, ce_a0r, ce_a0i, ce_a1r, ce_a1i, ce_pack;
    DeviceBuffer ldpc_bc, ldpc_shift, ldpc_degree, ldpc_edge;
    ce::LmmseWeightModelV1 ce_model{};

    explicit StaticWeights(const fs::path &root)
        : ofdm_w32r(ofdm::P * ofdm::P * kHalfBytes),
          ofdm_w32i(ofdm::P * ofdm::P * kHalfBytes),
          ofdm_w64r(ofdm::Q * ofdm::Q * kHalfBytes),
          ofdm_w64i(ofdm::Q * ofdm::Q * kHalfBytes),
          ofdm_twr(ofdm::P * ofdm::Q * kHalfBytes),
          ofdm_twi(ofdm::P * ofdm::Q * kHalfBytes),
          ce_br(ce::B_ELEMS * kHalfBytes), ce_bi(ce::B_ELEMS * kHalfBytes),
          ce_a0r(ce::A_ELEMS * kHalfBytes), ce_a0i(ce::A_ELEMS * kHalfBytes),
          ce_a1r(ce::A_ELEMS * kHalfBytes), ce_a1i(ce::WT_ELEMS * kHalfBytes),
          ce_pack(ce::PACK_INDEX_ELEMS * sizeof(uint32_t)),
          ldpc_bc(airan::PACKED_BYTES), ldpc_shift(airan::PACKED_BYTES),
          ldpc_degree(airan::LDPC_MB * sizeof(int16_t)),
          ldpc_edge((airan::LDPC_MB + 1) * sizeof(int32_t))
    {
        const fs::path ow = root / "weights/ofdm_demod";
        ofdm_w32r.Upload(ReadExact<uint16_t>(ow / "w_dft32_re.bin", ofdm::P * ofdm::P));
        ofdm_w32i.Upload(ReadExact<uint16_t>(ow / "w_dft32_im.bin", ofdm::P * ofdm::P));
        ofdm_w64r.Upload(ReadExact<uint16_t>(ow / "w_dft64_re_T.bin", ofdm::Q * ofdm::Q));
        ofdm_w64i.Upload(ReadExact<uint16_t>(ow / "w_dft64_im_T.bin", ofdm::Q * ofdm::Q));
        ofdm_twr.Upload(ReadExact<uint16_t>(ow / "twiddle_pq_re.bin", ofdm::P * ofdm::Q));
        ofdm_twi.Upload(ReadExact<uint16_t>(ow / "twiddle_pq_im.bin", ofdm::P * ofdm::Q));

        const fs::path cw = root / "weights/channel_est";
        ce_br.Upload(ReadExact<uint16_t>(cw / "factor_b_re.bin", ce::B_ELEMS));
        ce_bi.Upload(ReadExact<uint16_t>(cw / "factor_b_im.bin", ce::B_ELEMS));
        ce_a0r.Upload(ReadExact<uint16_t>(cw / "cube_time_fused_d0_re.bin", ce::A_ELEMS));
        ce_a0i.Upload(ReadExact<uint16_t>(cw / "cube_time_fused_d0_im.bin", ce::A_ELEMS));
        ce_a1r.Upload(ReadExact<uint16_t>(cw / "cube_time_fused_d1_re.bin", ce::A_ELEMS));
        ce_a1i.Upload(ReadExact<uint16_t>(cw / "cube_time_fused_d1_im.bin", ce::WT_ELEMS));
        ce_pack.Upload(ReadExact<uint32_t>(cw / "ce_pack_gather_index.bin", ce::PACK_INDEX_ELEMS));
        const auto model = ReadExact<ce::LmmseWeightModelV1>(cw / "weight_model.bin", 1);
        ce_model = model.front();

        const auto shift = ReadExact<int16_t>(root / "weights/ldpc_decode/shift_table.bin",
                                              airan::SHIFT_ELEMS);
        const auto degree = ReadExact<int16_t>(root / "weights/ldpc_decode/degrees.bin",
                                               airan::LDPC_MB);
        const auto edge = ReadExact<int32_t>(root / "weights/ldpc_decode/edge_offsets.bin",
                                             airan::LDPC_MB + 1);
        std::vector<int16_t> packed_bc(airan::PACKED_ELEMS_TOTAL, 0);
        std::vector<int16_t> packed_shift(airan::PACKED_ELEMS_TOTAL, 0);
        for (uint32_t row = 0; row < airan::LDPC_MB; ++row) {
            uint32_t packed = 0;
            for (uint32_t col = 0; col < airan::LDPC_NFULL; ++col) {
                const int16_t value = shift[row * airan::LDPC_NFULL + col];
                if (value < 0) continue;
                if (packed >= airan::LDPC_MAX_DEG) throw std::runtime_error("LDPC degree overflow");
                packed_bc[row * airan::LDPC_MAX_DEG + packed] = static_cast<int16_t>(col);
                packed_shift[row * airan::LDPC_MAX_DEG + packed] = value;
                ++packed;
            }
            if (packed != static_cast<uint32_t>(degree[row])) {
                throw std::runtime_error("LDPC degree/table mismatch");
            }
        }
        if (edge.back() != static_cast<int32_t>(airan::LDPC_TOTAL_EDGES)) {
            throw std::runtime_error("LDPC edge count mismatch");
        }
        ldpc_bc.Upload(packed_bc); ldpc_shift.Upload(packed_shift);
        ldpc_degree.Upload(degree); ldpc_edge.Upload(edge);
    }
};

int Run(const Options &options)
{
    const size_t iq_slot_elems = static_cast<size_t>(kRx) *
                                 ofdm::INPUT_INT16_PER_BATCH;
    const auto rx_iq = ReadExact<int16_t>(options.input_root / "rx_iq.bin",
                                          static_cast<size_t>(kSlots) * iq_slot_elems);

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    {
        airan::PuschMimoRuntimeConfig runtime{};
        airan::PuschMimoConfig config = LoadConfig(0, &runtime);
        StaticWeights weights(options.input_root);

        const size_t platform_workspace =
            platform_ascendc::PlatformAscendCManager::GetInstance(SOC_VERSION)
                ->GetLibApiWorkSpaceSize();
        DeviceBuffer workspace(std::max(kSharedWorkspaceBytes, platform_workspace));

        DeviceBuffer d_iq(iq_slot_elems * sizeof(int16_t));
        DeviceBuffer d_fft_re(rd::InputElems(kRx) * kHalfBytes);
        DeviceBuffer d_fft_im(rd::InputElems(kRx) * kHalfBytes);
        DeviceBuffer d_rx_re(rd::OutputElems(kRx) * kHalfBytes);
        DeviceBuffer d_rx_im(rd::OutputElems(kRx) * kHalfBytes);

        DeviceBuffer d_dmrs_re(dg::LogicalOutputElems(kRank, 2) * kHalfBytes);
        DeviceBuffer d_dmrs_im(dg::LogicalOutputElems(kRank, 2) * kHalfBytes);
        DeviceBuffer d_hls_re(ls::NaturalElems(kRank) * kHalfBytes);
        DeviceBuffer d_hls_im(ls::NaturalElems(kRank) * kHalfBytes);
        DeviceBuffer d_pilot_sc(static_cast<size_t>(kRank) * 2 * 832 * sizeof(uint16_t));
        DeviceBuffer d_pilot_count(ls::COUNT_PAD * sizeof(uint16_t));
        DeviceBuffer d_noise(ls::NR_CURRENT * kHalfBytes);

        DeviceBuffer d_ce_hr(ce::HLS_ELEMS * kHalfBytes);
        DeviceBuffer d_ce_hi(ce::HLS_ELEMS * kHalfBytes);
        DeviceBuffer d_ce_hn(ce::HLS_ELEMS * kHalfBytes);
        DeviceBuffer d_ce_tre(ce::T_ELEMS * kHalfBytes);
        DeviceBuffer d_ce_tim(ce::T_ELEMS * kHalfBytes);
        DeviceBuffer d_ce_out_re(ce::OUT_ELEMS * kHalfBytes);
        DeviceBuffer d_ce_out_im(ce::OUT_ELEMS * kHalfBytes);
        DeviceBuffer d_hgrid_re(ce::PADDED_OUT_ELEMS * kHalfBytes);
        DeviceBuffer d_hgrid_im(ce::PADDED_OUT_ELEMS * kHalfBytes);
        // The pad-layers adapter overwrites every active Rank4 plane.  Only
        // the inactive physical planes need initialization, so clear the
        // persistent detector tensor once rather than writing ~119 MiB in
        // every slot.
        ACL_CHECK(aclrtMemset(d_hgrid_re.get(), d_hgrid_re.bytes(), 0,
                              d_hgrid_re.bytes()));
        ACL_CHECK(aclrtMemset(d_hgrid_im.get(), d_hgrid_im.bytes(), 0,
                              d_hgrid_im.bytes()));

        DeviceBuffer d_hrm_re(io::PACKED_ELEMS * kHalfBytes);
        DeviceBuffer d_hrm_im(io::PACKED_ELEMS * kHalfBytes);
        DeviceBuffer d_grouped_y_re(kGroupedYElems * kHalfBytes);
        DeviceBuffer d_grouped_y_im(kGroupedYElems * kHalfBytes);
        DeviceBuffer d_no(io::NO_ELEMS * kHalfBytes);
        DeviceBuffer d_xhat_re(static_cast<size_t>(kLayerCapacity) * io::N_RE * kHalfBytes);
        DeviceBuffer d_xhat_im(static_cast<size_t>(kLayerCapacity) * io::N_RE * kHalfBytes);
        DeviceBuffer d_no_eff(static_cast<size_t>(kLayerCapacity) * io::N_RE * kHalfBytes);
        DeviceBuffer d_dummy(64);
        ACL_CHECK(aclrtMemset(d_dummy.get(), d_dummy.bytes(), 0, d_dummy.bytes()));

        DeviceBuffer d_layer_llr(layer::LayerElems(kRank) * sizeof(int16_t));
        const size_t cw_slot_elems = layer::CodewordElems(kRank);
        DeviceBuffer d_cw_llr(static_cast<size_t>(kSlots) * cw_slot_elems * sizeof(int16_t));

        DeviceBuffer d_ofdm_tiling(kOfdmTilingBytes);
        std::vector<uint8_t> ofdm_tiling(kOfdmTilingBytes, 0);
        GenerateOfdmDemodTiling(SOC_VERSION, ofdm_tiling.data());
        d_ofdm_tiling.Upload(ofdm_tiling);

        std::vector<uint32_t> gather(rd::N_SC_PAD);
        rd::KernelMetadata redemap_meta{};
        airan::PuschMimoLayout layout{};
        if (rd::BuildCurrentProfile(config, &layout, &redemap_meta) != rd::OK ||
            rd::BuildGatherIndex(gather.data(), gather.size()) != rd::OK) {
            throw std::runtime_error("RE demap profile failed");
        }
        DeviceBuffer d_gather(gather.size() * sizeof(uint32_t)); d_gather.Upload(gather);
        DeviceBuffer d_redemap_meta(sizeof(redemap_meta)); d_redemap_meta.Upload(&redemap_meta, sizeof(redemap_meta));

        dg::MimoDmrsGenRuntimeV1 *dmrs_runtime = nullptr;
        if (dg::CreateRuntime(&dmrs_runtime) != dg::OK || dmrs_runtime == nullptr) {
            throw std::runtime_error("DMRS runtime creation failed");
        }

        std::vector<uint16_t> expected_count(ls::COUNT_PAD, 0);
        std::vector<uint16_t> expected_sc(ls::MAX_LAYERS * 2 * 832, 0);
        ls::KernelMetadata ls_meta{};
        airan::PuschMimoLayout ls_layout{};
        if (ls::BuildCurrentProfile(config, &ls_layout, &ls_meta,
                                    expected_count.data(), expected_sc.data()) != ls::OK) {
            throw std::runtime_error("DMRS-LS profile failed");
        }
        DeviceBuffer d_ls_meta(ls::TILING_BYTES);
        d_ls_meta.Upload(&ls_meta, sizeof(ls_meta));

        std::vector<uint8_t> ce_tiling(ce::TILING_BYTES, 0);
        GenerateChannelEstLmmseTiling(SOC_VERSION, ce_tiling.data());
        DeviceBuffer d_ce_tiling(ce_tiling.size()); d_ce_tiling.Upload(ce_tiling);

        airan::MimoDetectLayerPlan detect_plan{};
        detect_plan.abi_version = airan::PUSCH_MIMO_ABI_VERSION;
        detect_plan.struct_size = sizeof(detect_plan);
        detect_plan.num_rx_antennas = kRx;
        detect_plan.layer_capacity = kLayerCapacity;
        detect_plan.total_layers = kRank;
        detect_plan.num_allocations = 1;
        detect_plan.allocations[0] = {0, 0, kRank, kRank};
        io::KernelMetadata io_meta{};
        if (io::BuildCurrentProfile(&config, 1, detect_plan, &io_meta) != io::OK) {
            throw std::runtime_error("detector IO profile failed");
        }
        DeviceBuffer d_io_meta(sizeof(io_meta)); d_io_meta.Upload(&io_meta, sizeof(io_meta));

        std::vector<uint8_t> bri_tiling(airan::TILING_TOTAL_SIZE, 0);
        GenerateMimoDetectBriTiling(SOC_VERSION, bri_tiling.data());
        DeviceBuffer d_bri_tiling(bri_tiling.size()); d_bri_tiling.Upload(bri_tiling);

        qam::BatchTilingData qam_meta{};
        airan::PuschMimoLayout qam_layout{};
        if (qam::BuildCurrentProfile(config, &qam_layout, &qam_meta) != qam::OK) {
            throw std::runtime_error("QAM profile failed");
        }
        DeviceBuffer d_qam_meta(sizeof(qam_meta)); d_qam_meta.Upload(&qam_meta, sizeof(qam_meta));

        std::vector<uint8_t> layer_tiling(layer::TILING_BYTES, 0);
        auto *layer_meta = reinterpret_cast<layer::KernelMetadata *>(layer_tiling.data());
        auto *layer_gather = reinterpret_cast<uint32_t *>(layer_tiling.data() + sizeof(*layer_meta));
        airan::PuschMimoLayout layer_layout{};
        if (layer::BuildCurrentProfile(config, &layer_layout, layer_meta, layer_gather) != layer::OK) {
            throw std::runtime_error("layer demap profile failed");
        }
        DeviceBuffer d_layer_tiling(layer_tiling.size()); d_layer_tiling.Upload(layer_tiling);

        std::map<std::string, std::vector<double>> metrics;
        std::map<std::string, EventTimer> timers;
        for (const char *name : {"slot_total", "ofdm_demod", "re_demap", "dmrs_gen",
                                 "dmrs_ls", "lmmse_ce", "io_pack_grouped", "bri_detect",
                                 "qam_demod", "layer_demap"}) {
            timers.try_emplace(name);
        }

        const auto wall_started = std::chrono::steady_clock::now();
        for (uint16_t slot = 0; slot < kSlots; ++slot) {
            config = LoadConfig(slot, &runtime);
            timers.at("slot_total").Begin(stream);
            ACL_CHECK(aclrtMemcpyAsync(d_iq.get(), d_iq.bytes(),
                                       rx_iq.data() + static_cast<size_t>(slot) * iq_slot_elems,
                                       d_iq.bytes(), ACL_MEMCPY_HOST_TO_DEVICE, stream));

            timers.at("ofdm_demod").Begin(stream);
            CheckLaunch(ACLRT_LAUNCH_KERNEL(ofdm_demod_batch_kernel)(
                ofdm::BLOCK_DIM, stream, d_iq.get(), weights.ofdm_w32r.get(),
                weights.ofdm_w32i.get(), weights.ofdm_w64r.get(), weights.ofdm_w64i.get(),
                weights.ofdm_twr.get(), weights.ofdm_twi.get(), d_fft_re.get(), d_fft_im.get(),
                kRx, workspace.get(), d_ofdm_tiling.get()), "ofdm_demod_batch");
            timers.at("ofdm_demod").End(stream);

            timers.at("re_demap").Begin(stream);
            rd::ReDemapBatchOpArgsV1 redemap_args{};
            redemap_args.abi_version = airan::PUSCH_MIMO_ABI_VERSION;
            redemap_args.struct_size = sizeof(redemap_args);
            redemap_args.fft_grid_re = d_fft_re.get(); redemap_args.fft_grid_im = d_fft_im.get();
            redemap_args.rx_grid_re = d_rx_re.get(); redemap_args.rx_grid_im = d_rx_im.get();
            redemap_args.config = &config; redemap_args.layout = &layout; redemap_args.stream = stream;
            if (rd::Enqueue(redemap_args, d_gather.get(), d_gather.bytes(), workspace.get(),
                            workspace.bytes(), d_redemap_meta.get(), d_redemap_meta.bytes()) != rd::OK) {
                throw std::runtime_error("re_demap enqueue failed");
            }
            timers.at("re_demap").End(stream);

            timers.at("dmrs_gen").Begin(stream);
            dg::MimoDmrsGenOpArgsV1 dmrs_args{};
            dmrs_args.abi_version = airan::PUSCH_MIMO_ABI_VERSION;
            dmrs_args.struct_size = sizeof(dmrs_args);
            dmrs_args.dmrs_re = d_dmrs_re.get(); dmrs_args.dmrs_im = d_dmrs_im.get();
            dmrs_args.config = &config; dmrs_args.layout = &layout; dmrs_args.stream = stream;
            if (dg::Enqueue(dmrs_runtime, dmrs_args) != dg::OK) {
                throw std::runtime_error("DMRS enqueue failed");
            }
            timers.at("dmrs_gen").End(stream);

            timers.at("dmrs_ls").Begin(stream);
            CheckLaunch(ACLRT_LAUNCH_KERNEL(mimo_dmrs_ls_kernel)(
                ls::BLOCK_DIM, stream, d_rx_re.get(), d_rx_im.get(), d_dmrs_re.get(),
                d_dmrs_im.get(), d_hls_re.get(), d_hls_im.get(), d_pilot_sc.get(),
                d_pilot_count.get(), d_noise.get(), workspace.get(), d_ls_meta.get()),
                "mimo_dmrs_ls");
            timers.at("dmrs_ls").End(stream);

            ce::ChannelEstLmmseOpArgsV1 ce_args{};
            ce_args.abi_version = airan::PUSCH_MIMO_ABI_VERSION;
            ce_args.struct_size = sizeof(ce_args);
            ce_args.h_ls_re = d_hls_re.get(); ce_args.h_ls_im = d_hls_im.get();
            ce_args.pilot_count = d_pilot_count.get(); ce_args.pilot_sc = d_pilot_sc.get();
            ce_args.pilot_count_host = expected_count.data(); ce_args.pilot_sc_host = expected_sc.data();
            ce_args.weight_model = &weights.ce_model;
            ce_args.h_grid_re = d_hgrid_re.get(); ce_args.h_grid_im = d_hgrid_im.get();
            ce_args.config = &config; ce_args.layout = &ls_layout; ce_args.stream = stream;
            if (ce::ValidateOpArgs(ce_args) != ce::OK) throw std::runtime_error("CE contract failed");
            timers.at("lmmse_ce").Begin(stream);
            CheckLaunch(ACLRT_LAUNCH_KERNEL(channel_est_lmmse_pack_natural_kernel)(
                ce::BLOCK_DIM, stream, d_hls_re.get(), d_hls_im.get(), weights.ce_pack.get(),
                d_ce_hr.get(), d_ce_hi.get(), d_ce_hn.get(), workspace.get(), d_ce_tiling.get()),
                "channel_est_lmmse_pack_natural");
            ACL_CHECK(aclrtMemsetAsync(d_ce_tre.get(), d_ce_tre.bytes(), 0, d_ce_tre.bytes(), stream));
            ACL_CHECK(aclrtMemsetAsync(d_ce_tim.get(), d_ce_tim.bytes(), 0, d_ce_tim.bytes(), stream));
            ACL_CHECK(aclrtMemsetAsync(workspace.get(), workspace.bytes(), 0, 4u * 1024u * 1024u, stream));
            CheckLaunch(ACLRT_LAUNCH_KERNEL(channel_est_lmmse_kernel)(
                ce::BLOCK_DIM, stream, weights.ce_br.get(), weights.ce_bi.get(),
                d_ce_hr.get(), d_ce_hi.get(), d_ce_hn.get(), weights.ce_a0r.get(),
                weights.ce_a0i.get(), weights.ce_a1r.get(), weights.ce_a1i.get(),
                weights.ce_a1i.get(), d_ce_tre.get(), d_ce_tim.get(), d_ce_out_re.get(),
                d_ce_out_im.get(), workspace.get(), d_ce_tiling.get()), "channel_est_lmmse");
            CheckLaunch(ACLRT_LAUNCH_KERNEL(channel_est_lmmse_pad_layers_kernel)(
                ce::BLOCK_DIM, stream, d_ce_out_re.get(), d_ce_out_im.get(),
                d_hgrid_re.get(), d_hgrid_im.get(), workspace.get(), d_ce_tiling.get()),
                "channel_est_lmmse_pad_layers");
            timers.at("lmmse_ce").End(stream);

            timers.at("io_pack_grouped").Begin(stream);
            io::MimoDetectIoPackOpArgsV1 io_args{};
            io_args.abi_version = airan::PUSCH_MIMO_ABI_VERSION;
            io_args.struct_size = sizeof(io_args);
            io_args.rx_grid_re = d_rx_re.get(); io_args.rx_grid_im = d_rx_im.get();
            io_args.h_grid_re = d_hgrid_re.get(); io_args.h_grid_im = d_hgrid_im.get();
            io_args.noise_var_rx = d_noise.get(); io_args.hrm_re = d_hrm_re.get();
            io_args.hrm_im = d_hrm_im.get(); io_args.yvpad_re = d_grouped_y_re.get();
            io_args.yvpad_im = d_grouped_y_im.get(); io_args.no = d_no.get();
            io_args.configs = &config; io_args.num_configs = 1;
            io_args.layer_plan = &detect_plan; io_args.stream = stream;
            if (io::Enqueue(io_args, d_io_meta.get(), d_io_meta.bytes()) != io::OK) {
                throw std::runtime_error("fused grouped IO pack failed");
            }
            timers.at("io_pack_grouped").End(stream);

            timers.at("bri_detect").Begin(stream);
            CheckLaunch(ACLRT_LAUNCH_KERNEL(mimo_detect_bri_kernel)(
                airan::BLOCK_DIM, stream, d_hrm_re.get(), d_hrm_im.get(),
                d_dummy.get(), d_dummy.get(), d_no.get(), d_dummy.get(),
                d_xhat_re.get(), d_xhat_im.get(), d_no_eff.get(),
                d_grouped_y_re.get(), d_grouped_y_im.get(), workspace.get(),
                d_bri_tiling.get()), "mimo_detect_bri");
            timers.at("bri_detect").End(stream);

            timers.at("qam_demod").Begin(stream);
            qam::QamDemod256BatchOpArgsV1 qam_args{};
            qam_args.abi_version = airan::PUSCH_MIMO_ABI_VERSION;
            qam_args.struct_size = sizeof(qam_args);
            qam_args.x_re = d_xhat_re.get(); qam_args.x_im = d_xhat_im.get();
            qam_args.no_eff = d_no_eff.get(); qam_args.layer_llr = d_layer_llr.get();
            qam_args.config = &config; qam_args.layout = &qam_layout; qam_args.stream = stream;
            if (qam::Enqueue(qam_args, workspace.get(), workspace.bytes(),
                             d_qam_meta.get(), d_qam_meta.bytes()) != qam::OK) {
                throw std::runtime_error("QAM demod enqueue failed");
            }
            timers.at("qam_demod").End(stream);

            timers.at("layer_demap").Begin(stream);
            CheckLaunch(ACLRT_LAUNCH_KERNEL(layer_demap_kernel)(
                layer::BLOCK_DIM, stream, d_layer_llr.get(),
                d_cw_llr.offset(static_cast<size_t>(slot) * cw_slot_elems * sizeof(int16_t)),
                workspace.get(), d_layer_tiling.get()), "layer_demap");
            timers.at("layer_demap").End(stream);
            timers.at("slot_total").End(stream);

            ACL_CHECK(aclrtSynchronizeStream(stream));
            for (auto &[name, timer] : timers) AppendMetric(&metrics, name, &timer);
        }

        dg::DestroyRuntime(dmrs_runtime);

        config = LoadConfig(0, &runtime);
        descr::KernelMetadata descr_meta{};
        airan::PuschMimoLayout descr_layout{};
        if (descr::BuildCurrentProfile(config, kSlots, &descr_layout, &descr_meta) != descr::OK) {
            throw std::runtime_error("descramble profile failed");
        }
        const size_t coded_elems = descr::BufferElems(kSlots, kRank);
        std::vector<int16_t> signs(coded_elems);
        if (descr::BuildGoldSign(config, descr_layout, kSlots, signs.data(), signs.size()) != descr::OK) {
            throw std::runtime_error("Gold sign generation failed");
        }
        DeviceBuffer d_sign(signs.size() * sizeof(int16_t)); d_sign.Upload(signs);
        DeviceBuffer d_llr_nr(coded_elems * sizeof(int16_t));
        DeviceBuffer d_descr_meta(sizeof(descr_meta)); d_descr_meta.Upload(&descr_meta, sizeof(descr_meta));

        rate::KernelMetadata rate_meta{};
        airan::PuschMimoLayout rate_layout{};
        std::vector<rate::RateMatchDescriptor> descriptors(rate::DESC_PAD_WORDS / rate::DESC_WORDS);
        if (rate::BuildCurrentProfile(config, kSlots, &rate_layout, &rate_meta,
                                      descriptors.data(), rate::C_NUM) != rate::OK) {
            throw std::runtime_error("rate-dematch profile failed");
        }
        DeviceBuffer d_descriptors(rate::DESCRIPTOR_BYTES); d_descriptors.Upload(descriptors.data(), rate::DESCRIPTOR_BYTES);
        DeviceBuffer d_rate_meta(sizeof(rate_meta)); d_rate_meta.Upload(&rate_meta, sizeof(rate_meta));
        DeviceBuffer d_ldpc_llr(rate::OutputElems() * sizeof(int16_t));

        DeviceBuffer d_prev(airan::PREV_BYTES), d_bits(airan::BITS_BYTES);
        DeviceBuffer d_lam_out(airan::LAM_BYTES), d_ldpc_scratch(airan::LAM_SCRATCH_BYTES);
        ACL_CHECK(aclrtMemset(d_prev.get(), d_prev.bytes(), 0, d_prev.bytes()));
        ACL_CHECK(aclrtMemset(d_bits.get(), d_bits.bytes(), 0xff, d_bits.bytes()));
        ACL_CHECK(aclrtMemset(d_lam_out.get(), d_lam_out.bytes(), 0, d_lam_out.bytes()));
        ACL_CHECK(aclrtMemset(d_ldpc_scratch.get(), d_ldpc_scratch.bytes(), 0, d_ldpc_scratch.bytes()));

        EventTimer descr_timer, rate_timer, ldpc_timer, tb_timer;
        tb_timer.Begin(stream);
        descr_timer.Begin(stream);
        descr::DescrambleMimoOpArgsV1 descr_args{};
        descr_args.abi_version = airan::PUSCH_MIMO_ABI_VERSION;
        descr_args.struct_size = sizeof(descr_args);
        descr_args.llr_qam = d_cw_llr.get(); descr_args.llr_nr = d_llr_nr.get();
        descr_args.config = &config; descr_args.layout = &descr_layout;
        descr_args.num_slots = kSlots; descr_args.stream = stream;
        if (descr::Enqueue(descr_args, d_sign.get(), d_sign.bytes(), workspace.get(),
                           workspace.bytes(), d_descr_meta.get(), d_descr_meta.bytes()) != descr::OK) {
            throw std::runtime_error("descramble enqueue failed");
        }
        descr_timer.End(stream);

        rate_timer.Begin(stream);
        rate::RateDematchMimoOpArgsV1 rate_args{};
        rate_args.abi_version = airan::PUSCH_MIMO_ABI_VERSION;
        rate_args.struct_size = sizeof(rate_args);
        rate_args.cw_llr = d_llr_nr.get(); rate_args.ldpc_llr = d_ldpc_llr.get();
        rate_args.config = &config; rate_args.layout = &rate_layout;
        rate_args.num_slots = kSlots; rate_args.stream = stream;
        if (rate::Enqueue(rate_args, d_descriptors.get(), d_descriptors.bytes(), workspace.get(),
                          workspace.bytes(), d_rate_meta.get(), d_rate_meta.bytes()) != rate::OK) {
            throw std::runtime_error("rate-dematch enqueue failed");
        }
        rate_timer.End(stream);

        ldpc_timer.Begin(stream);
        CheckLaunch(ACLRT_LAUNCH_KERNEL(ldpc_decode_kernel)(
            4, stream, d_ldpc_llr.get(), weights.ldpc_bc.get(), weights.ldpc_shift.get(),
            weights.ldpc_degree.get(), weights.ldpc_edge.get(), d_prev.get(), d_bits.get(),
            d_lam_out.get(), d_ldpc_scratch.get()), "ldpc_decode");
        ldpc_timer.End(stream); tb_timer.End(stream);
        ACL_CHECK(aclrtSynchronizeStream(stream));

        metrics["descramble"].push_back(descr_timer.ReadUs());
        metrics["rate_dematch"].push_back(rate_timer.ReadUs());
        metrics["ldpc_decode"].push_back(ldpc_timer.ReadUs());
        metrics["tb_tail"].push_back(tb_timer.ReadUs());

        std::vector<int8_t> bits(airan::BITS_BYTES);
        ACL_CHECK(aclrtMemcpy(bits.data(), bits.size(), d_bits.get(), d_bits.bytes(),
                              ACL_MEMCPY_DEVICE_TO_HOST));
        if (!std::all_of(bits.begin(), bits.end(), [](int8_t bit) { return bit == 0 || bit == 1; })) {
            throw std::runtime_error("LDPC output is not binary");
        }
        WriteExact(options.output, bits);

        long long bit_errors = -1;
        const fs::path expected_path = options.input_root / "expected_bits.bin";
        if (fs::is_regular_file(expected_path)) {
            const auto expected = ReadExact<int8_t>(expected_path, bits.size());
            bit_errors = 0;
            for (size_t i = 0; i < bits.size(); ++i) bit_errors += bits[i] != expected[i];
            if (bit_errors != 0) throw std::runtime_error("decoded information BER is nonzero");
        }

        const double wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wall_started).count();
        fs::create_directories(options.receipt.parent_path());
        std::ofstream receipt(options.receipt, std::ios::trunc);
        receipt << "{\n"
                << "  \"schema\": \"airan.pusch_mimo.fused_rx.v1\",\n"
                << "  \"status\": \"PASS\",\n"
                << "  \"rank\": 4,\n"
                << "  \"tx_antennas\": 16,\n"
                << "  \"rx_antennas\": 64,\n"
                << "  \"slots\": 23,\n"
                << "  \"single_acl_context\": true,\n"
                << "  \"single_stream\": true,\n"
                << "  \"intermediate_d2h\": false,\n"
                << "  \"grouped_rhs_on_device\": true,\n"
                << "  \"wall_ms\": " << wall_ms << ",\n"
                << "  \"bit_errors\": " << bit_errors << ",\n"
                << "  \"p50_us\": {\n";
        size_t emitted = 0;
        for (const auto &[name, samples] : metrics) {
            receipt << "    \"" << name << "\": " << Median(samples)
                    << (++emitted == metrics.size() ? "\n" : ",\n");
        }
        receipt << "  }\n}\n";
        if (!receipt) throw std::runtime_error("receipt write failed");

        std::printf("==================================================================\n");
        std::printf(" AI-RAN NPU PUSCH MIMO RX — fused 16TX x 64RX Rank4\n");
        std::printf("==================================================================\n");
        for (const auto &[name, samples] : metrics) {
            std::printf(" %-24s p50 %10.1f us\n", name.c_str(), Median(samples));
        }
        std::printf(" wall 23 slots + TB tail  %10.3f ms\n", wall_ms);
        std::printf(" decoded bits=%zu errors=%lld\n", bits.size(), bit_errors);
        std::printf("==================================================================\n");
    }

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}

}  // namespace

int main(int argc, char **argv)
{
    try {
        return Run(ParseOptions(argc, argv));
    } catch (const std::exception &error) {
        std::fprintf(stderr, "[HARD_FAIL] %s\n", error.what());
        return 1;
    }
}
