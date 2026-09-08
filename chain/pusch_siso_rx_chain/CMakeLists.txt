cmake_minimum_required(VERSION 3.16)
project(rx_chain)

# ── 与各 kernel 的 CMakeLists 同构(复制自 ofdm_demod 模板)───────────────
set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend310P1" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "ASCEND CANN package installation directory")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "" FORCE)
endif()

set(KERNEL_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)   # kernels

# NOTE: descramble 故意不在 RX_KERNEL_DIRS。它被 18+ kernel 合并 device 链接搞坏
# (aicore exception 0x26;standalone 单 kernel 链接 bit-exact)。
# 改为下方单独的 ascendc_kernels_descr lib —— 单 kernel device image,与 standalone 一致。
set(RX_KERNEL_DIRS
    sync/decimate
    sync/pss_correlator
    sync/pss_cfo_estimator
    sync/sss_correlator
    sync/ssb_fft
    sync/pbch_dmrs_correlator
    sync/cfo_estimate
    sync/cfo_compensate
    ofdm/ofdm_demod_siso
    mapping/re_demap_siso
    channel_est/channel_est_ls_siso
    equalization/equalize_zf_siso
    sync/cfo_dmrs
    sync/timing_tracker
    fec/rate_dematch_siso
    fec/ldpc_decode
    sync/win_slice_siso)

# Production sources are explicit so an experimental translation unit cannot
# enter the clean chain through a glob.
set(KERNEL_FILES
    ${KERNEL_ROOT}/sync/decimate/decimate_kernel.cpp
    ${KERNEL_ROOT}/sync/pss_correlator/pss_correlator_kernel.cpp
    ${KERNEL_ROOT}/sync/pss_cfo_estimator/pss_cfo_estimator_kernel.cpp
    ${KERNEL_ROOT}/sync/sss_correlator/sss_correlator_kernel.cpp
    ${KERNEL_ROOT}/sync/ssb_fft/ssb_fft_kernel.cpp
    ${KERNEL_ROOT}/sync/pbch_dmrs_correlator/pbch_dmrs_correlator_kernel.cpp
    ${KERNEL_ROOT}/sync/cfo_estimate/cfo_estimate_kernel.cpp
    ${KERNEL_ROOT}/sync/cfo_compensate/cfo_compensate_kernel.cpp
    ${KERNEL_ROOT}/ofdm/ofdm_demod_siso/ofdm_demod_kernel.cpp
    ${KERNEL_ROOT}/mapping/re_demap_siso/re_demap_kernel.cpp
    ${KERNEL_ROOT}/channel_est/channel_est_ls_siso/channel_est_ls_kernel.cpp
    ${KERNEL_ROOT}/equalization/equalize_zf_siso/equalize_kernel.cpp
    ${KERNEL_ROOT}/sync/cfo_dmrs/cfo_dmrs_kernel.cpp
    ${KERNEL_ROOT}/sync/timing_tracker/timing_tracker_kernel.cpp
    ${KERNEL_ROOT}/fec/rate_dematch_siso/rate_dematch_kernel.cpp
    ${KERNEL_ROOT}/fec/ldpc_decode/ldpc_decode_kernel.cpp
    ${KERNEL_ROOT}/sync/win_slice_siso/win_slice_kernel.cpp)

# Canonical QAM demod outputs the standard chain ABI [8,12,1600].
list(APPEND KERNEL_FILES
    ${KERNEL_ROOT}/mapping/qam_demod_256_siso/qam256_demod_kernel.cpp)

set(CUSTOM_ASCEND310P_LIST "Ascend310P1" "Ascend310P3")

# cmake/{cpu_lib,npu_lib}.cmake 复制自任一 kernel 目录(它们一致)
# npu_lib.cmake 内部 include(ascendc.cmake) 并 ascendc_library(ascendc_kernels_npu ...)
if("${RUN_MODE}" STREQUAL "cpu")
    include(cmake/cpu_lib.cmake)
elseif("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
    include(cmake/npu_lib.cmake)
endif()

# ── descramble 独立 lib(单 kernel device image)─────────────────────────
# ascendc_library 已被 npu_lib.cmake 里的 include(ascendc.cmake) 定义好,这里第二次调用。
# 内部 target 都带 lib 名前缀(ascendc_kernels_descr_aic_device 等),与主 lib 不撞名。
# 镜像 npu_lib.cmake 对主 lib 的编译定义。
if("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
    ascendc_library(ascendc_kernels_descr SHARED
        ${KERNEL_ROOT}/fec/descramble_siso/descramble_kernel.cpp)
    ascendc_compile_definitions(ascendc_kernels_descr PRIVATE
        $<$<BOOL:$<IN_LIST:${SOC_VERSION},${CUSTOM_ASCEND310P_LIST}>>:CUSTOM_ASCEND310P>
        HAVE_WORKSPACE
        HAVE_TILING
    )
endif()

# Host tiling sources are explicit as well; a stray generated/experimental shim
# cannot silently enter the production executable.
set(SHIM_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/tiling_shims/cfo_compensate_shim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tiling_shims/cfo_dmrs_shim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tiling_shims/cfo_estimate_shim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tiling_shims/channel_est_ls_shim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tiling_shims/decimate_shim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tiling_shims/descramble_shim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tiling_shims/equalize_shim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tiling_shims/ldpc_decode_shim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tiling_shims/ofdm_demod_shim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tiling_shims/pss_cfo_estimator_shim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tiling_shims/pss_correlator_shim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tiling_shims/qam256_demod_shim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tiling_shims/rate_dematch_shim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tiling_shims/rx_tiling_setup.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tiling_shims/ssb_fft_shim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tiling_shims/timing_tracker_shim.cpp)

add_executable(rx_chain
    ${CMAKE_CURRENT_SOURCE_DIR}/rx_chain.cpp
    ${KERNEL_ROOT}/sync/sss_correlator/sss_correlator_tiling.cpp   # GenerateSssTiling (host-filled tiling, 唯一符号)
    ${SHIM_FILES}
)

# 各 op 目录入口(供 <op>.h / tiling.cpp 内部 include),+ 本目录
target_include_directories(rx_chain PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
foreach(op_dir ${RX_KERNEL_DIRS})
    target_include_directories(rx_chain PRIVATE ${KERNEL_ROOT}/${op_dir})
endforeach()
# descramble 单独补 include 目录（shim 相对 include + descramble.h）。
target_include_directories(rx_chain PRIVATE ${KERNEL_ROOT}/fec/descramble_siso)

target_compile_options(rx_chain PRIVATE
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall
)

target_compile_definitions(rx_chain PRIVATE
    $<$<BOOL:$<IN_LIST:${SOC_VERSION},${CUSTOM_ASCEND310P_LIST}>>:CUSTOM_ASCEND310P>
    SOC_VERSION="${SOC_VERSION}"
)

# 同时链主 lib 和 descramble 独立 lib;descr 的 aclrtlaunch_descramble_kernel.h
# 经由 target 的 interface include 传递给 rx_chain。
target_link_libraries(rx_chain PRIVATE
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
    ascendc_kernels_${RUN_MODE}
    ascendc_kernels_descr
    tiling_api register platform ascendalog dl
)

install(TARGETS rx_chain RUNTIME DESTINATION bin)
