#include "tx_chain.h"

#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "aclrtlaunch_ldpc_encode_kernel.h"
#include "aclrtlaunch_layer_map_kernel.h"
#include "aclrtlaunch_mimo_resource_grid_map_kernel.h"
#include "aclrtlaunch_ofdm_mod_batch_kernel.h"
#include "kernel_tiling/kernel_tiling.h"
#include "ldpc_encode.h"
#include "layer_map.h"
#include "mimo_dmrs_gen.h"
#include "mimo_resource_grid_map.h"
#include "ofdm_mod.h"
#include "pusch_codebook_precode_runtime.h"
#include "pusch_mimo_runtime_config.h"
#include "qam_mod_256_mimo.h"
#include "rate_match_mimo.h"
#include "re_map_batch.h"
#include "scramble_mimo.h"
#include "tiling/platform/platform_ascendc.h"

extern "C" void BuildMimoTxOfdmTiling(const char *soc_version, uint8_t *buffer);

namespace fs = std::filesystem;

namespace airan::pusch_mimo_tx_chain {
namespace {

namespace rm = airan::rate_match_mimo;
namespace sm = airan::scramble_mimo;
namespace qm = airan::qam_mod_256_mimo;
namespace lm = airan::layer_map;
namespace dg = airan::mimo_dmrs_gen;
namespace gm = airan::mimo_resource_grid_map;
namespace pc = airan::pusch_precode;
namespace re = airan::re_map_batch;
namespace ofdm = ofdm_mod_batch;

constexpr uint32_t kBlockDim = 4;
constexpr size_t kInfoFourBytes = 4u * airan::LDPC_K;
constexpr size_t kDebugBytes =
    static_cast<size_t>(airan::LDPC_C_NUM) * airan::DBG_WORDS_PER_CB *
    sizeof(int32_t);

[[noreturn]] void Fail(const std::string &message)
{
    throw std::runtime_error(message);
}

void CheckAcl(aclError status, const char *expression)
{
    if (status != ACL_ERROR_NONE) {
        Fail(std::string(expression) + " returned " + std::to_string(status));
    }
}

#define ACL_CHECK(expression) CheckAcl((expression), #expression)

void CheckStatus(bool ok, const char *stage)
{
    if (!ok) Fail(std::string(stage) + " rejected the shared chain contract");
}

class AclSession {
public:
    AclSession()
    {
        ACL_CHECK(aclInit(nullptr));
        initialized_ = true;
        ACL_CHECK(aclrtSetDevice(0));
        device_set_ = true;
        ACL_CHECK(aclrtCreateStream(&stream_));
    }

    ~AclSession()
    {
        if (stream_ != nullptr) aclrtDestroyStream(stream_);
        if (device_set_) aclrtResetDevice(0);
        if (initialized_) aclFinalize();
    }

    aclrtStream stream() const { return stream_; }

private:
    aclrtStream stream_ = nullptr;
    bool initialized_ = false;
    bool device_set_ = false;
};

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t bytes) { Allocate(bytes); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    ~DeviceBuffer()
    {
        if (data_ != nullptr) aclrtFree(data_);
    }

    void Allocate(size_t bytes)
    {
        if (data_ != nullptr || bytes == 0) Fail("invalid device allocation");
        ACL_CHECK(aclrtMalloc(&data_, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        bytes_ = bytes;
        ACL_CHECK(aclrtMemset(data_, bytes_, 0, bytes_));
    }

    void Upload(const void *source, size_t bytes)
    {
        if (source == nullptr || bytes > bytes_) Fail("device upload exceeds buffer");
        ACL_CHECK(aclrtMemcpy(data_, bytes_, source, bytes,
                             ACL_MEMCPY_HOST_TO_DEVICE));
    }

    void Download(void *destination, size_t bytes) const
    {
        if (destination == nullptr || bytes > bytes_) Fail("device download exceeds buffer");
        ACL_CHECK(aclrtMemcpy(destination, bytes, data_, bytes,
                             ACL_MEMCPY_DEVICE_TO_HOST));
    }

    void *get() const { return data_; }
    size_t bytes() const { return bytes_; }

    void *offset(size_t bytes) const
    {
        if (bytes > bytes_) Fail("device pointer offset exceeds buffer");
        return static_cast<uint8_t *>(data_) + bytes;
    }

private:
    void *data_ = nullptr;
    size_t bytes_ = 0;
};

std::vector<uint8_t> ReadExact(const fs::path &path, size_t bytes)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || static_cast<size_t>(input.tellg()) != bytes) {
        Fail(path.string() + " expected exactly " + std::to_string(bytes) + " bytes");
    }
    std::vector<uint8_t> result(bytes);
    input.seekg(0);
    input.read(reinterpret_cast<char *>(result.data()),
               static_cast<std::streamsize>(bytes));
    if (!input) Fail("short read from " + path.string());
    return result;
}

void WriteExact(const fs::path &path, const void *data, size_t bytes)
{
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
    if (!output) Fail("cannot write " + path.string());
}

PuschMimoConfig LegacyConfig(uint16_t rank)
{
    PuschMimoConfig config{};
    config.abi_version = PUSCH_MIMO_ABI_VERSION;
    config.struct_size = sizeof(config);
    config.num_layers = rank;
    config.num_tx_ports = rank == 1 ? 1 : (rank == 2 ? 2 : 4);
    config.num_rx_antennas = 64;
    config.qm = 8;
    config.num_symbols = 14;
    config.fft_size = 2048;
    config.num_rb = 133;
    config.num_allocated_symbols = 14;
    config.used_subcarriers = 1596;
    config.padded_subcarriers = 1664;
    config.dmrs_symbol_mask = static_cast<uint16_t>((1u << 2) | (1u << 11));
    const uint16_t ports[4] = {1000, 1001, 1002, 1003};
    for (uint16_t layer = 0; layer < rank; ++layer) {
        config.dmrs_ports[layer] =
            rank == 2 && layer == 1 ? 1002 : ports[layer];
    }
    config.dmrs_type = 1;
    config.dmrs_length = 1;
    config.num_cdm_groups_without_data = 2;
    config.codebook_enabled = rank == 3 ? 1 : 0;
    config.tpmi = rank == 3 ? 6 : 0;
    config.prg_size_rb = rank == 3 ? 4 : 133;
    return config;
}

PuschMimoConfig ResolveConfig(const RunOptions &options, bool *profile_enabled,
                              PuschMimoRuntimeConfig *runtime_out)
{
    PuschMimoRuntimeConfig runtime{};
    std::string why;
    if (LoadMimoRuntimeConfigFromEnv(options.rank, &runtime, profile_enabled, &why) !=
        MimoRuntimeConfigStatus::kSuccess) {
        Fail("runtime profile: " + why);
    }
    PuschMimoConfig config{};
    if (*profile_enabled) {
        if (runtime.num_layers > MAX_LAYERS ||
            runtime.num_tx_ports > MAX_LOGICAL_PORTS ||
            runtime.num_tx_antennas > MAX_TX_ANTENNAS ||
            runtime.qm != 8 || runtime.num_symbols != 14 ||
            runtime.fft_size != 2048 || runtime.used_subcarriers != 1596 ||
            runtime.padded_subcarriers != 1664 || runtime.num_dmrs_symbols != 2) {
            Fail("runtime profile exceeds the current fused TX execution shape");
        }
        if (DerivePuschMimoConfig(runtime, runtime.max_rx_antennas, &config, &why) !=
            MimoRuntimeConfigStatus::kSuccess) {
            Fail("derive operator config: " + why);
        }
    } else {
        config = LegacyConfig(options.rank);
    }
    if (!*profile_enabled || options.cell_id_override) {
        config.data_scrambling_id = options.data_scrambling_id;
        config.dmrs_scrambling_id = options.dmrs_scrambling_id;
    }
    if (!*profile_enabled || options.rnti_override) config.rnti = options.rnti;
    if (runtime_out != nullptr) *runtime_out = runtime;
    return config;
}

rm::RateMatchFecConfigV1 FecConfig()
{
    rm::RateMatchFecConfigV1 fec{};
    fec.abi_version = rm::ABI_VERSION;
    fec.struct_size = sizeof(fec);
    fec.num_code_blocks = rm::C_NUM;
    fec.encoded_stride = rm::N_CB_BUF;
    fec.base_graph = 1;
    fec.lifting_size = rm::LDPC_Z;
    return fec;
}

template <typename T>
DeviceBuffer UploadObject(const T &value)
{
    DeviceBuffer buffer(sizeof(value));
    buffer.Upload(&value, sizeof(value));
    return buffer;
}

// DeviceBuffer is deliberately non-movable; explicit helpers keep ownership
// visible at every chain resource boundary.
void UploadPadded(DeviceBuffer &destination, const fs::path &path,
                  size_t logical_bytes)
{
    const auto value = ReadExact(path, logical_bytes);
    destination.Upload(value.data(), logical_bytes);
}

struct StaticWeights {
    DeviceBuffer shift_a;
    DeviceBuffer shift_bi;
    DeviceBuffer shift_c;
    DeviceBuffer shift_d;
    DeviceBuffer w32r;
    DeviceBuffer w32i;
    DeviceBuffer w64r;
    DeviceBuffer w64i;
    DeviceBuffer twr;
    DeviceBuffer twi;

    explicit StaticWeights(const fs::path &root)
        : shift_a(((airan::SHIFT_A_ELEMS + 15) / 16) * 16 * sizeof(int16_t)),
          shift_bi(((airan::SHIFT_BI_ELEMS + 15) / 16) * 16 * sizeof(int16_t)),
          shift_c(((airan::SHIFT_C_ELEMS + 15) / 16) * 16 * sizeof(int16_t)),
          shift_d(((airan::SHIFT_D_ELEMS + 15) / 16) * 16 * sizeof(int16_t)),
          w32r(static_cast<size_t>(ofdm::P) * ofdm::P * sizeof(int16_t)),
          w32i(static_cast<size_t>(ofdm::P) * ofdm::P * sizeof(int16_t)),
          w64r(static_cast<size_t>(ofdm::Q) * ofdm::Q * sizeof(int16_t)),
          w64i(static_cast<size_t>(ofdm::Q) * ofdm::Q * sizeof(int16_t)),
          twr(static_cast<size_t>(ofdm::P) * ofdm::Q * sizeof(int16_t)),
          twi(static_cast<size_t>(ofdm::P) * ofdm::Q * sizeof(int16_t))
    {
        const fs::path ldpc = root / "weights/ldpc_bg1_z384_shifts";
        UploadPadded(shift_a, ldpc / "shift_A.bin", airan::SHIFT_A_BYTES);
        UploadPadded(shift_bi, ldpc / "shift_Bi.bin", airan::SHIFT_BI_BYTES);
        UploadPadded(shift_c, ldpc / "shift_C.bin", airan::SHIFT_C_BYTES);
        UploadPadded(shift_d, ldpc / "shift_D.bin", airan::SHIFT_D_BYTES);
        const fs::path ofdm_root = root / "weights/ofdm";
        UploadPadded(w32r, ofdm_root / "iw_dft32_re.bin", w32r.bytes());
        UploadPadded(w32i, ofdm_root / "iw_dft32_im.bin", w32i.bytes());
        UploadPadded(w64r, ofdm_root / "iw_dft64_re_T.bin", w64r.bytes());
        UploadPadded(w64i, ofdm_root / "iw_dft64_im_T.bin", w64i.bytes());
        UploadPadded(twr, ofdm_root / "itwiddle_pq_re.bin", twr.bytes());
        UploadPadded(twi, ofdm_root / "itwiddle_pq_im.bin", twi.bytes());
    }
};

int RunImpl(const RunOptions &options)
{
    if (options.input_root == nullptr || options.output_iq == nullptr ||
        options.rank < 1 || options.rank > MAX_LAYERS ||
        options.num_slots == 0 || options.num_slots > NUM_SLOTS) {
        Fail("input root, output path, Rank1..4 and slots1..23 are required");
    }
    const fs::path input_root(options.input_root);
    bool profile_enabled = false;
    PuschMimoRuntimeConfig runtime{};
    PuschMimoConfig config = ResolveConfig(options, &profile_enabled, &runtime);
    const uint16_t layers = config.num_layers;
    const uint16_t logical_ports = config.num_tx_ports;
    const uint16_t tx_antennas = profile_enabled
        ? runtime.num_tx_antennas : logical_ports;
    const uint32_t slots = options.num_slots;

    AclSession session;
    StaticWeights weights(input_root);

    // Tile the four deterministic LDPC reference blocks to the production
    // 143-code-block transport batch, matching the canonical encoder runner.
    const auto four_info = ReadExact(input_root / "golden/input.bin", kInfoFourBytes);
    std::vector<uint8_t> info(airan::INFO_BYTES);
    for (uint32_t cb = 0; cb < airan::LDPC_C_NUM; ++cb) {
        std::memcpy(info.data() + static_cast<size_t>(cb) * airan::LDPC_K,
                    four_info.data() + static_cast<size_t>(cb % 4) * airan::LDPC_K,
                    airan::LDPC_K);
    }

    DeviceBuffer d_info(airan::INFO_BYTES);
    DeviceBuffer d_encoded(airan::OUTPUT_BYTES);
    DeviceBuffer d_ldpc_debug(kDebugBytes);
    d_info.Upload(info.data(), info.size());

    const size_t codeword_elems = rm::OutputElems(slots, layers);
    const size_t codeword_bytes = codeword_elems * sizeof(int16_t);
    DeviceBuffer d_bits_nr(codeword_bytes);
    DeviceBuffer d_bits_qam(codeword_bytes);

    const auto fec = FecConfig();
    PuschMimoLayout layout{};
    rm::KernelMetadata rate_meta{};
    CheckStatus(rm::BuildCurrentProfile(config, fec, slots, &layout, &rate_meta) == rm::OK,
                "rate_match_mimo profile");
    std::vector<rm::RateMatchDescriptor> descriptors(rm::C_NUM);
    CheckStatus(rm::BuildRateMatchDescriptors(config, fec, slots, descriptors.data(),
                                               descriptors.size()) == rm::OK,
                "rate_match_mimo descriptors");
    DeviceBuffer d_descriptors(descriptors.size() * sizeof(descriptors[0]));
    d_descriptors.Upload(descriptors.data(), d_descriptors.bytes());
    DeviceBuffer d_rate_meta(sizeof(rate_meta));
    d_rate_meta.Upload(&rate_meta, sizeof(rate_meta));

    sm::KernelMetadata scramble_meta{};
    PuschMimoLayout scramble_layout{};
    CheckStatus(sm::BuildCurrentProfile(config, slots, &scramble_layout,
                                        &scramble_meta) == sm::OK,
                "scramble_mimo profile");
    std::vector<int16_t> gold(codeword_elems);
    CheckStatus(sm::BuildGoldBits(config, scramble_layout, slots, gold.data(),
                                  gold.size()) == sm::OK,
                "scramble_mimo gold");
    DeviceBuffer d_gold(codeword_bytes);
    d_gold.Upload(gold.data(), codeword_bytes);
    DeviceBuffer d_scramble_meta(sizeof(scramble_meta));
    d_scramble_meta.Upload(&scramble_meta, sizeof(scramble_meta));

    qm::KernelMetadata qam_meta{};
    PuschMimoLayout qam_layout{};
    CheckStatus(qm::BuildCurrentProfile(config, &qam_layout, &qam_meta) == qm::OK,
                "qam_mod_256_mimo profile");
    DeviceBuffer d_qam_meta(sizeof(qam_meta));
    d_qam_meta.Upload(&qam_meta, sizeof(qam_meta));

    lm::KernelMetadata layer_meta{};
    PuschMimoLayout layer_layout{};
    std::vector<uint32_t> gather(lm::MAX_INDEX_ELEMS);
    CheckStatus(lm::BuildCurrentProfile(config, &layer_layout, &layer_meta,
                                        gather.data()) == lm::OK,
                "layer_map_mimo profile");
    std::vector<uint8_t> layer_tiling(lm::TILING_BYTES, 0);
    std::memcpy(layer_tiling.data(), &layer_meta, sizeof(layer_meta));
    std::memcpy(layer_tiling.data() + sizeof(layer_meta), gather.data(),
                gather.size() * sizeof(gather[0]));
    DeviceBuffer d_layer_tiling(layer_tiling.size());
    d_layer_tiling.Upload(layer_tiling.data(), layer_tiling.size());

    gm::KernelMetadata grid_meta{};
    PuschMimoLayout grid_layout{};
    std::vector<uint32_t> data_offsets(gm::N_DATA_PAD);
    std::vector<uint32_t> dmrs_offsets(gm::DmrsOffsetElems());
    CheckStatus(gm::BuildCurrentProfile(config, &grid_layout, &grid_meta,
                                        data_offsets.data(), dmrs_offsets.data()) == gm::OK,
                "mimo_resource_grid_map profile");
    DeviceBuffer d_data_offsets(data_offsets.size() * sizeof(data_offsets[0]));
    DeviceBuffer d_dmrs_offsets(dmrs_offsets.size() * sizeof(dmrs_offsets[0]));
    DeviceBuffer d_grid_meta(sizeof(grid_meta));
    d_data_offsets.Upload(data_offsets.data(), d_data_offsets.bytes());
    d_dmrs_offsets.Upload(dmrs_offsets.data(), d_dmrs_offsets.bytes());
    d_grid_meta.Upload(&grid_meta, sizeof(grid_meta));

    PuschMimoConfig antenna_config = config;
    if (profile_enabled) {
        antenna_config.num_layers = logical_ports;
        antenna_config.num_tx_ports = tx_antennas;
        antenna_config.codebook_enabled = static_cast<uint8_t>(
            MimoPrecodingMode::kNonCodebook);
        antenna_config.tpmi = 0;
        antenna_config.prg_size_rb = pc::N_RB;
    }

    PuschMimoLayout remap_layout{};
    std::vector<uint32_t> scatter(re::SCATTER_INDEX_ELEMS);
    CheckStatus(re::BuildCurrentProfile(antenna_config, &remap_layout, scatter.data(),
                                        scatter.size()) == re::OK,
                "re_map_batch profile");
    DeviceBuffer d_scatter(scatter.size() * sizeof(scatter[0]));
    d_scatter.Upload(scatter.data(), d_scatter.bytes());

    const size_t workspace_bytes = std::max<size_t>(
        16u * 1024u * 1024u,
        platform_ascendc::PlatformAscendCManager::GetInstance(SOC_VERSION)
            ->GetLibApiWorkSpaceSize());
    DeviceBuffer d_workspace(workspace_bytes);
    DeviceBuffer d_dummy(re::DUMMY_TILING_BYTES);

    constexpr size_t ofdm_tiling_bytes = 2 * sizeof(TCubeTiling);
    std::vector<uint8_t> ofdm_tiling(ofdm_tiling_bytes, 0);
    BuildMimoTxOfdmTiling(SOC_VERSION, ofdm_tiling.data());
    DeviceBuffer d_ofdm_tiling(ofdm_tiling.size());
    d_ofdm_tiling.Upload(ofdm_tiling.data(), ofdm_tiling.size());

    DeviceBuffer d_qam_re(static_cast<size_t>(layers) * N_DATA_PAD * sizeof(uint16_t));
    DeviceBuffer d_qam_im(static_cast<size_t>(layers) * N_DATA_PAD * sizeof(uint16_t));
    DeviceBuffer d_layer_re(static_cast<size_t>(layers) * N_DATA_PAD * sizeof(uint16_t));
    DeviceBuffer d_layer_im(static_cast<size_t>(layers) * N_DATA_PAD * sizeof(uint16_t));
    DeviceBuffer d_dmrs_re(gm::DmrsElems(layers) * sizeof(uint16_t));
    DeviceBuffer d_dmrs_im(gm::DmrsElems(layers) * sizeof(uint16_t));
    DeviceBuffer d_layer_grid_re(gm::GridElems(layers) * sizeof(uint16_t));
    DeviceBuffer d_layer_grid_im(gm::GridElems(layers) * sizeof(uint16_t));
    DeviceBuffer d_port_grid_re(re::PortGridElems(logical_ports) * sizeof(uint16_t));
    DeviceBuffer d_port_grid_im(re::PortGridElems(logical_ports) * sizeof(uint16_t));
    DeviceBuffer d_antenna_grid_re(re::PortGridElems(tx_antennas) * sizeof(uint16_t));
    DeviceBuffer d_antenna_grid_im(re::PortGridElems(tx_antennas) * sizeof(uint16_t));
    DeviceBuffer d_fft_grid_re(re::FftGridElems(tx_antennas) * sizeof(uint16_t));
    DeviceBuffer d_fft_grid_im(re::FftGridElems(tx_antennas) * sizeof(uint16_t));
    DeviceBuffer d_time_re(static_cast<size_t>(tx_antennas) * N_TIME_SAMPLES * sizeof(int16_t));
    DeviceBuffer d_time_im(static_cast<size_t>(tx_antennas) * N_TIME_SAMPLES * sizeof(int16_t));
    const size_t iq_slot_bytes =
        static_cast<size_t>(tx_antennas) * N_TIME_SAMPLES * 2 * sizeof(int16_t);
    const size_t iq_total_bytes = static_cast<size_t>(slots) * iq_slot_bytes;
    DeviceBuffer d_iq(iq_total_bytes);

    dg::MimoDmrsGenRuntimeV1 *dmrs_runtime = nullptr;
    CheckStatus(dg::CreateRuntime(&dmrs_runtime) == dg::OK && dmrs_runtime != nullptr,
                "mimo_dmrs_gen runtime");
    pc::RuntimeContext precode_runtime{};
    CheckStatus(pc::RuntimeInit(&precode_runtime) == pc::Status::kSuccess,
                "pusch_codebook_precode runtime");

    const auto started = std::chrono::steady_clock::now();
    const uint32_t ldpc_launch = ACLRT_LAUNCH_KERNEL(ldpc_encode_kernel)(
        kBlockDim, session.stream(), d_info.get(), weights.shift_a.get(),
        weights.shift_bi.get(), weights.shift_c.get(), weights.shift_d.get(),
        d_ldpc_debug.get(), d_encoded.get());
    CheckStatus(ldpc_launch == ACL_ERROR_NONE, "ldpc_encode launch");

    rm::RateMatchMimoOpArgsV1 rate_args{};
    rate_args.abi_version = rm::ABI_VERSION;
    rate_args.struct_size = sizeof(rate_args);
    rate_args.code_blocks = d_encoded.get();
    rate_args.bits_nr = d_bits_nr.get();
    rate_args.config = &config;
    rate_args.layout = &layout;
    rate_args.fec = &fec;
    rate_args.num_slots = slots;
    rate_args.stream = session.stream();
    CheckStatus(rm::Enqueue(rate_args, d_descriptors.get(), d_descriptors.bytes(),
                            d_workspace.get(), d_workspace.bytes(),
                            d_rate_meta.get(), d_rate_meta.bytes()) == rm::OK,
                "rate_match_mimo launch");

    sm::ScrambleMimoOpArgsV1 scramble_args{};
    scramble_args.abi_version = sm::ABI_VERSION;
    scramble_args.struct_size = sizeof(scramble_args);
    scramble_args.bits_nr = d_bits_nr.get();
    scramble_args.bits_qam = d_bits_qam.get();
    scramble_args.config = &config;
    scramble_args.layout = &scramble_layout;
    scramble_args.num_slots = slots;
    scramble_args.stream = session.stream();
    CheckStatus(sm::Enqueue(scramble_args, d_gold.get(), d_gold.bytes(),
                            d_workspace.get(), d_workspace.bytes(),
                            d_scramble_meta.get(), d_scramble_meta.bytes()) == sm::OK,
                "scramble_mimo launch");

    for (uint16_t slot = 0; slot < slots; ++slot) {
        config.slot_number = slot;
        const size_t slot_bit_offset =
            static_cast<size_t>(slot) * 8u * layers * N_DATA_PAD * sizeof(int16_t);

        qm::QamMod256MimoOpArgsV1 qam_args{};
        qam_args.abi_version = qm::ABI_VERSION;
        qam_args.struct_size = sizeof(qam_args);
        qam_args.bits_qam = d_bits_qam.offset(slot_bit_offset);
        qam_args.d_re = d_qam_re.get();
        qam_args.d_im = d_qam_im.get();
        qam_args.config = &config;
        qam_args.layout = &qam_layout;
        qam_args.stream = session.stream();
        CheckStatus(qm::Enqueue(qam_args, d_workspace.get(), d_workspace.bytes(),
                                d_qam_meta.get(), d_qam_meta.bytes()) == qm::OK,
                    "qam_mod_256_mimo launch");

        const uint32_t layer_launch = ACLRT_LAUNCH_KERNEL(layer_map_kernel)(
            lm::BLOCK_DIM, session.stream(), d_qam_re.get(), d_qam_im.get(),
            d_layer_re.get(), d_layer_im.get(), d_workspace.get(),
            d_layer_tiling.get());
        CheckStatus(layer_launch == ACL_ERROR_NONE, "layer_map_mimo launch");

        dg::MimoDmrsGenOpArgsV1 dmrs_args{};
        dmrs_args.abi_version = dg::ABI_VERSION;
        dmrs_args.struct_size = sizeof(dmrs_args);
        dmrs_args.dmrs_re = d_dmrs_re.get();
        dmrs_args.dmrs_im = d_dmrs_im.get();
        dmrs_args.config = &config;
        dmrs_args.layout = &grid_layout;
        dmrs_args.stream = session.stream();
        CheckStatus(dg::Enqueue(dmrs_runtime, dmrs_args) == dg::OK,
                    "mimo_dmrs_gen launch");

        const uint32_t grid_launch =
            ACLRT_LAUNCH_KERNEL(mimo_resource_grid_map_kernel)(
                gm::BLOCK_DIM, session.stream(), d_layer_re.get(), d_layer_im.get(),
                d_dmrs_re.get(), d_dmrs_im.get(), d_data_offsets.get(),
                d_dmrs_offsets.get(), d_layer_grid_re.get(), d_layer_grid_im.get(),
                d_workspace.get(), d_grid_meta.get());
        CheckStatus(grid_launch == ACL_ERROR_NONE,
                    "mimo_resource_grid_map launch");

        std::string why;
        CheckStatus(pc::Launch(&precode_runtime, d_layer_grid_re.get(),
                               d_layer_grid_im.get(), config,
                               d_port_grid_re.get(), d_port_grid_im.get(),
                               d_workspace.get(), session.stream(), &why) ==
                        pc::Status::kSuccess,
                    ("pusch_codebook_precode launch: " + why).c_str());

        const void *air_grid_re = d_port_grid_re.get();
        const void *air_grid_im = d_port_grid_im.get();
        if (profile_enabled) {
            why.clear();
            CheckStatus(pc::Launch(&precode_runtime, d_port_grid_re.get(),
                                   d_port_grid_im.get(), antenna_config,
                                   d_antenna_grid_re.get(), d_antenna_grid_im.get(),
                                   d_workspace.get(), session.stream(), &why) ==
                            pc::Status::kSuccess,
                        ("tx_antenna_map launch: " + why).c_str());
            air_grid_re = d_antenna_grid_re.get();
            air_grid_im = d_antenna_grid_im.get();
        }

        re::ReMapBatchOpArgsV1 remap_args{};
        remap_args.abi_version = re::ABI_VERSION;
        remap_args.struct_size = sizeof(remap_args);
        remap_args.port_grid_re = air_grid_re;
        remap_args.port_grid_im = air_grid_im;
        remap_args.fft_grid_re = d_fft_grid_re.get();
        remap_args.fft_grid_im = d_fft_grid_im.get();
        remap_args.config = &antenna_config;
        remap_args.layout = &remap_layout;
        remap_args.stream = session.stream();
        CheckStatus(re::Enqueue(remap_args, d_scatter.get(), d_scatter.bytes(),
                                d_workspace.get(), d_workspace.bytes(),
                                d_dummy.get(), d_dummy.bytes()) == re::OK,
                    "re_map_batch launch");

        const uint32_t ofdm_launch = ACLRT_LAUNCH_KERNEL(ofdm_mod_batch_kernel)(
            ofdm::BLOCK_DIM, session.stream(), d_fft_grid_re.get(),
            d_fft_grid_im.get(), weights.w32r.get(), weights.w32i.get(),
            weights.w64r.get(), weights.w64i.get(), weights.twr.get(),
            weights.twi.get(), d_time_re.get(), d_time_im.get(),
            d_iq.offset(static_cast<size_t>(slot) * iq_slot_bytes), tx_antennas,
            d_workspace.get(), d_ofdm_tiling.get());
        CheckStatus(ofdm_launch == ACL_ERROR_NONE, "ofdm_mod_batch launch");

        // DMRS runtime reuses pinned host metadata on the next Enqueue. This
        // slot boundary is the only synchronization inside the fused chain.
        ACL_CHECK(aclrtSynchronizeStream(session.stream()));
    }

    std::vector<int16_t> iq(iq_total_bytes / sizeof(int16_t));
    d_iq.Download(iq.data(), iq_total_bytes);
    if (std::none_of(iq.begin(), iq.end(), [](int16_t value) { return value != 0; })) {
        Fail("fused TX output is all zero");
    }
    WriteExact(options.output_iq, iq.data(), iq_total_bytes);

    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    if (options.receipt_json != nullptr) {
        const size_t nonzero = static_cast<size_t>(std::count_if(
            iq.begin(), iq.end(), [](int16_t value) { return value != 0; }));
        std::string json =
            "{\n  \"schema\": \"airan.pusch_mimo.tx_chain.v1\",\n" +
            std::string("  \"rank\": ") + std::to_string(layers) + ",\n" +
            "  \"logical_tx_ports\": " + std::to_string(logical_ports) + ",\n" +
            "  \"tx_antennas\": " + std::to_string(tx_antennas) + ",\n" +
            "  \"slots\": " + std::to_string(slots) + ",\n" +
            "  \"output_bytes\": " + std::to_string(iq_total_bytes) + ",\n" +
            "  \"nonzero_i16\": " + std::to_string(nonzero) + ",\n" +
            "  \"single_acl_context\": true,\n" +
            "  \"shared_device_arena\": true,\n" +
            "  \"intermediate_host_files\": false,\n" +
            "  \"profile_enabled\": " +
                std::string(profile_enabled ? "true" : "false") + ",\n" +
            "  \"elapsed_ms\": " + std::to_string(elapsed) + ",\n" +
            "  \"status\": \"PASS\"\n}\n";
        WriteExact(options.receipt_json, json.data(), json.size());
    }

    pc::RuntimeDestroy(&precode_runtime);
    dg::DestroyRuntime(dmrs_runtime);
    std::printf("[PASS] fused MIMO TX Rank%u logical_ports=%u tx_antennas=%u slots=%u bytes=%zu %.3f ms profile=%s\n",
                layers, logical_ports, tx_antennas, slots, iq_total_bytes, elapsed,
                profile_enabled ? "runtime" : "legacy");
    return 0;
}

}  // namespace

int Run(const RunOptions &options)
{
    try {
        return RunImpl(options);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "[HARD_FAIL] %s\n", error.what());
        return 1;
    }
}

}  // namespace airan::pusch_mimo_tx_chain

int main(int argc, char **argv)
{
    using airan::pusch_mimo_tx_chain::RunOptions;
    RunOptions options{};
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        auto require_value = [&](const char *name) -> const char * {
            if (++index >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(2);
            }
            return argv[index];
        };
        if (argument == "--input-root") options.input_root = require_value("--input-root");
        else if (argument == "--output") options.output_iq = require_value("--output");
        else if (argument == "--receipt") options.receipt_json = require_value("--receipt");
        else if (argument == "--rank") {
            options.rank = static_cast<uint16_t>(std::strtoul(require_value("--rank"), nullptr, 10));
        } else if (argument == "--slots") {
            options.num_slots = static_cast<uint16_t>(std::strtoul(require_value("--slots"), nullptr, 10));
        } else if (argument == "--cell-id") {
            const auto value = static_cast<uint16_t>(std::strtoul(require_value("--cell-id"), nullptr, 10));
            options.data_scrambling_id = value;
            options.dmrs_scrambling_id = value;
            options.cell_id_override = true;
        } else if (argument == "--rnti") {
            options.rnti = static_cast<uint16_t>(std::strtoul(require_value("--rnti"), nullptr, 10));
            options.rnti_override = true;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argument.c_str());
            return 2;
        }
    }
    if (options.input_root == nullptr || options.output_iq == nullptr) {
        std::fprintf(stderr,
            "usage: %s --input-root DIR --output FILE --rank 1..4 "
            "[--slots 1..23] [--receipt FILE] [--cell-id N] [--rnti N]\n",
            argv[0]);
        return 2;
    }
    return airan::pusch_mimo_tx_chain::Run(options);
}
