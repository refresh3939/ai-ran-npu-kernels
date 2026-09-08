cmake_minimum_required(VERSION 3.16)
project(tx_chain)

# ── 与各 kernel 的 CMakeLists 同构(复制自 rx_chain 模板)───────────────
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

# Production SISO PUSCH sources are explicit.  Experiments and SSB interpolate
# are deliberately outside this device image.
set(TX_KERNEL_DIRS
    fec/ldpc_encode
    fec/rate_match_siso
    fec/scramble_siso
    mapping/qam_mod_256_siso
    sync/dmrs_gen_siso
    mapping/merge_dmrs_siso
    mapping/re_map_siso
    ofdm/ofdm_mod_siso)
set(KERNEL_FILES
    ${KERNEL_ROOT}/fec/ldpc_encode/ldpc_encode_kernel.cpp
    ${KERNEL_ROOT}/fec/rate_match_siso/rate_match_kernel.cpp
    ${KERNEL_ROOT}/fec/scramble_siso/scramble_kernel.cpp
    ${KERNEL_ROOT}/mapping/qam_mod_256_siso/qam256_mod_kernel.cpp
    ${KERNEL_ROOT}/sync/dmrs_gen_siso/dmrs_gen_kernel.cpp
    ${KERNEL_ROOT}/mapping/merge_dmrs_siso/merge_dmrs_kernel.cpp
    ${KERNEL_ROOT}/mapping/re_map_siso/re_map_kernel.cpp
    ${KERNEL_ROOT}/ofdm/ofdm_mod_siso/ofdm_mod_kernel.cpp)

set(CUSTOM_ASCEND310P_LIST "Ascend310P1" "Ascend310P3")

# cmake/{cpu_lib,npu_lib}.cmake 复制自任一 kernel 目录(它们一致)
if("${RUN_MODE}" STREQUAL "cpu")
    include(cmake/cpu_lib.cmake)
elseif("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
    include(cmake/npu_lib.cmake)
endif()

# Host tiling sources are explicit for the same provenance reason.
set(SHIM_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/tiling_shims/ldpc_encode_shim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tiling_shims/rate_match_shim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tiling_shims/scramble_shim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tiling_shims/qam256_mod_shim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tiling_shims/dmrs_gen_shim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tiling_shims/tx_tiling_setup.cpp)

add_executable(tx_chain
    ${CMAKE_CURRENT_SOURCE_DIR}/tx_chain.cpp
    ${SHIM_FILES}
)

# 各 child-op include 目录和本 chain 目录。
target_include_directories(tx_chain PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
foreach(op_dir ${TX_KERNEL_DIRS})
    target_include_directories(tx_chain PRIVATE ${KERNEL_ROOT}/${op_dir})
endforeach()

target_compile_options(tx_chain PRIVATE
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall
)

target_compile_definitions(tx_chain PRIVATE
    $<$<BOOL:$<IN_LIST:${SOC_VERSION},${CUSTOM_ASCEND310P_LIST}>>:CUSTOM_ASCEND310P>
    SOC_VERSION="${SOC_VERSION}"
)

target_link_libraries(tx_chain PRIVATE
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
    ascendc_kernels_${RUN_MODE}
    tiling_api register platform ascendalog dl
)

install(TARGETS tx_chain RUNTIME DESTINATION bin)
