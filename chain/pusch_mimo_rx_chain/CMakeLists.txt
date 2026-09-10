cmake_minimum_required(VERSION 3.16)
project(pusch_mimo_rx_chain LANGUAGES C CXX)

set(RUN_MODE "npu" CACHE STRING "sim/npu")
set(SOC_VERSION "Ascend310P1" CACHE STRING "target SoC")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "Ascend CANN package")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/install"
        CACHE STRING "" FORCE)
endif()
if(NOT RUN_MODE STREQUAL "npu" AND NOT RUN_MODE STREQUAL "sim")
    message(FATAL_ERROR "pusch_mimo_rx_chain supports RUN_MODE=npu|sim")
endif()

set(KERNEL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(RX_OPERATOR_DIRS
    ofdm/ofdm_demod_batch
    mapping/re_demap_batch
    mimo/mimo_dmrs_gen
    channel_est/mimo_dmrs_ls
    channel_est/channel_est_lmmse_mimo
    mimo/mimo_detect_io_pack
    mimo/mimo_detect_bri_batch
    mapping/qam_demod_256_mimo_batch
    mapping/qam_demod_256_siso
    mapping/layer_demap_mimo
    fec/descramble_mimo
    fec/rate_dematch_mimo
    fec/ldpc_decode)

set(KERNEL_FILES
    "${KERNEL_ROOT}/ofdm/ofdm_demod_batch/ofdm_demod_kernel.cpp"
    "${KERNEL_ROOT}/mapping/re_demap_batch/re_demap_batch_kernel.cpp"
    "${KERNEL_ROOT}/mimo/mimo_dmrs_gen/mimo_dmrs_gen_kernel.cpp"
    "${KERNEL_ROOT}/channel_est/mimo_dmrs_ls/mimo_dmrs_ls_kernel.cpp"
    "${KERNEL_ROOT}/channel_est/channel_est_lmmse_mimo/channel_est_lmmse_kernel.cpp"
    "${KERNEL_ROOT}/channel_est/channel_est_lmmse_mimo/channel_est_lmmse_adapters_kernel.cpp"
    "${KERNEL_ROOT}/mimo/mimo_detect_io_pack/mimo_detect_io_pack_kernel.cpp"
    "${KERNEL_ROOT}/mimo/mimo_detect_bri_batch/mimo_detect_bri_kernel.cpp"
    "${KERNEL_ROOT}/mapping/qam_demod_256_mimo_batch/qam256_demod_batch_kernel.cpp"
    "${KERNEL_ROOT}/mapping/layer_demap_mimo/layer_demap_kernel.cpp"
    "${KERNEL_ROOT}/fec/descramble_mimo/descramble_mimo_kernel.cpp"
    "${KERNEL_ROOT}/fec/rate_dematch_mimo/rate_dematch_mimo_kernel.cpp"
    "${KERNEL_ROOT}/fec/ldpc_decode/ldpc_decode_kernel.cpp")

foreach(relative_dir IN LISTS RX_OPERATOR_DIRS)
    include_directories("${KERNEL_ROOT}/${relative_dir}")
endforeach()
include_directories("${KERNEL_ROOT}/common")

include("${KERNEL_ROOT}/fec/rate_dematch_mimo/cmake/npu_lib.cmake")

set(RX_PROFILE_DEFINITIONS
    CUSTOM_ASCEND310P
    HAVE_WORKSPACE
    HAVE_TILING
    MIMO_DMRS_LS_NR=64
    MIMO_DMRS_LS_BLOCK_DIM=4
    CE_NR=64
    CE_NL=4
    CE_RANK=96
    CE_BLOCK_DIM=4
    CE_CUBE_TIME_FUSED=1
    MIMO_DETECT_IO_NR=64
    MIMO_DETECT_IO_NL=16
    AIRAN_FUSED_RX_GROUPED_Y=1
    MIMO_M=64
    MIMO_K=16
    MIMO_KR=4
    BRI_B=8
    AIRAN_PACK=0
    N_SC_PAD_VAL=1664
    GROUP_BATCH=8
    RE_WINDOW=5824
    N_WINDOW_PER_CORE=1
    BRI_GROUPED_NR=64
    BRI_GROUPED_NL=16)

target_compile_definitions(ascendc_kernels_${RUN_MODE} PRIVATE
    ${RX_PROFILE_DEFINITIONS})
if(RUN_MODE STREQUAL "npu" OR RUN_MODE STREQUAL "sim")
    ascendc_compile_definitions(ascendc_kernels_${RUN_MODE} PRIVATE
        ${RX_PROFILE_DEFINITIONS})
    ascendc_compile_options(ascendc_kernels_${RUN_MODE} PRIVATE
        -forward-options-to-host-compiler
        -DMIMO_DMRS_LS_NR=64 -DMIMO_DMRS_LS_BLOCK_DIM=4
        -DCE_NR=64 -DCE_NL=4 -DCE_RANK=96 -DCE_BLOCK_DIM=4
        -DCE_CUBE_TIME_FUSED=1
        -DMIMO_DETECT_IO_NR=64 -DMIMO_DETECT_IO_NL=16
        -DAIRAN_FUSED_RX_GROUPED_Y=1
        -DMIMO_M=64 -DMIMO_K=16 -DMIMO_KR=4 -DBRI_B=8
        -DAIRAN_PACK=0 -DN_SC_PAD_VAL=1664
        -DGROUP_BATCH=8 -DRE_WINDOW=5824 -DN_WINDOW_PER_CORE=1
        -DBRI_GROUPED_NR=64 -DBRI_GROUPED_NL=16)
endif()

add_executable(pusch_mimo_rx_chain
    rx_chain.cpp
    tiling_shims/ofdm_demod_batch_shim.cpp
    tiling_shims/channel_est_lmmse_shim.cpp
    tiling_shims/mimo_detect_bri_shim.cpp
    "${KERNEL_ROOT}/common/pusch_mimo_runtime_config.cpp"
    "${KERNEL_ROOT}/mapping/re_demap_batch/re_demap_batch.cpp"
    "${KERNEL_ROOT}/mimo/mimo_dmrs_gen/mimo_dmrs_gen.cpp"
    "${KERNEL_ROOT}/mimo/mimo_dmrs_gen/mimo_dmrs_gen_runtime.cpp"
    "${KERNEL_ROOT}/channel_est/mimo_dmrs_ls/mimo_dmrs_ls.cpp"
    "${KERNEL_ROOT}/channel_est/channel_est_lmmse_mimo/channel_est_lmmse.cpp"
    "${KERNEL_ROOT}/mimo/mimo_detect_io_pack/mimo_detect_io_pack.cpp"
    "${KERNEL_ROOT}/mimo/mimo_detect_io_pack/mimo_detect_io_pack_runtime.cpp"
    "${KERNEL_ROOT}/mapping/qam_demod_256_mimo_batch/qam256_demod_batch.cpp"
    "${KERNEL_ROOT}/mapping/qam_demod_256_mimo_batch/qam256_demod_batch_contract.cpp"
    "${KERNEL_ROOT}/mapping/layer_demap_mimo/layer_demap.cpp"
    "${KERNEL_ROOT}/fec/descramble_mimo/descramble_mimo.cpp"
    "${KERNEL_ROOT}/fec/rate_dematch_mimo/rate_dematch_mimo.cpp")

set_source_files_properties(rx_chain.cpp PROPERTIES COMPILE_OPTIONS "-std=c++17")
target_include_directories(pusch_mimo_rx_chain PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${KERNEL_ROOT}/common")
foreach(relative_dir IN LISTS RX_OPERATOR_DIRS)
    target_include_directories(pusch_mimo_rx_chain PRIVATE
        "${KERNEL_ROOT}/${relative_dir}")
endforeach()
target_compile_features(pusch_mimo_rx_chain PRIVATE cxx_std_17)
target_compile_options(pusch_mimo_rx_chain PRIVATE
    -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror)
target_compile_definitions(pusch_mimo_rx_chain PRIVATE
    ${RX_PROFILE_DEFINITIONS} SOC_VERSION="${SOC_VERSION}")
target_link_libraries(pusch_mimo_rx_chain PRIVATE
    host_intf_pub ascendc_kernels_${RUN_MODE}
    tiling_api register platform ascendalog c_sec dl)

install(TARGETS pusch_mimo_rx_chain RUNTIME DESTINATION bin)
