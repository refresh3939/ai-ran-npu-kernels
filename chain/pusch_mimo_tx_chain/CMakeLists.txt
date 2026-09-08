cmake_minimum_required(VERSION 3.16)
project(pusch_mimo_tx_chain LANGUAGES C CXX)

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
    message(FATAL_ERROR "pusch_mimo_tx_chain supports RUN_MODE=npu|sim")
endif()

set(KERNEL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(TX_OPERATOR_DIRS
    fec/ldpc_encode
    fec/rate_match_mimo
    fec/scramble_mimo
    mapping/qam_mod_256_mimo
    mapping/layer_map_mimo
    mimo/mimo_dmrs_gen
    mapping/mimo_resource_grid_map
    mimo/pusch_codebook_precode
    mapping/re_map_batch
    mapping/re_map_siso
    ofdm/ofdm_mod_batch)

# Production sources are intentionally explicit. No source glob and no
# experiment directory may enter the device image.
set(KERNEL_FILES
    "${KERNEL_ROOT}/fec/ldpc_encode/ldpc_encode_kernel.cpp"
    "${KERNEL_ROOT}/fec/rate_match_mimo/rate_match_mimo_kernel.cpp"
    "${KERNEL_ROOT}/fec/scramble_mimo/scramble_mimo_kernel.cpp"
    "${KERNEL_ROOT}/mapping/qam_mod_256_mimo/qam_mod_256_mimo_kernel.cpp"
    "${KERNEL_ROOT}/mapping/layer_map_mimo/layer_map_kernel.cpp"
    "${KERNEL_ROOT}/mimo/mimo_dmrs_gen/mimo_dmrs_gen_kernel.cpp"
    "${KERNEL_ROOT}/mapping/mimo_resource_grid_map/mimo_resource_grid_map_kernel.cpp"
    "${KERNEL_ROOT}/mimo/pusch_codebook_precode/pusch_codebook_precode_kernel.cpp"
    "${KERNEL_ROOT}/mapping/re_map_siso/re_map_kernel.cpp"
    "${KERNEL_ROOT}/ofdm/ofdm_mod_batch/ofdm_mod_kernel.cpp")

foreach(relative_dir IN LISTS TX_OPERATOR_DIRS)
    include_directories("${KERNEL_ROOT}/${relative_dir}")
endforeach()
include_directories("${KERNEL_ROOT}/common")

include("${KERNEL_ROOT}/fec/rate_match_mimo/cmake/npu_lib.cmake")

add_executable(pusch_mimo_tx_chain
    tx_chain.cpp
    tiling_shims/ofdm_mod_batch_shim.cpp
    "${KERNEL_ROOT}/common/pusch_mimo_runtime_config.cpp"
    "${KERNEL_ROOT}/fec/rate_match_mimo/rate_match_mimo.cpp"
    "${KERNEL_ROOT}/fec/scramble_mimo/scramble_mimo.cpp"
    "${KERNEL_ROOT}/mapping/qam_mod_256_mimo/qam_mod_256_mimo.cpp"
    "${KERNEL_ROOT}/mapping/layer_map_mimo/layer_map.cpp"
    "${KERNEL_ROOT}/mimo/mimo_dmrs_gen/mimo_dmrs_gen.cpp"
    "${KERNEL_ROOT}/mimo/mimo_dmrs_gen/mimo_dmrs_gen_runtime.cpp"
    "${KERNEL_ROOT}/mapping/mimo_resource_grid_map/mimo_resource_grid_map.cpp"
    "${KERNEL_ROOT}/mimo/pusch_codebook_precode/pusch_codebook_precode.cpp"
    "${KERNEL_ROOT}/mimo/pusch_codebook_precode/pusch_codebook_precode_runtime.cpp"
    "${KERNEL_ROOT}/mapping/re_map_batch/re_map_batch.cpp")

# The Ascend host interface exports -std=c++11. Put the chain translation
# unit's required standard at source scope so it remains the final -std flag.
set_source_files_properties(tx_chain.cpp PROPERTIES COMPILE_OPTIONS "-std=c++17")

target_include_directories(pusch_mimo_tx_chain PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${KERNEL_ROOT}/common")
foreach(relative_dir IN LISTS TX_OPERATOR_DIRS)
    target_include_directories(pusch_mimo_tx_chain PRIVATE
        "${KERNEL_ROOT}/${relative_dir}")
endforeach()

target_compile_features(pusch_mimo_tx_chain PRIVATE cxx_std_17)
target_compile_options(pusch_mimo_tx_chain PRIVATE
    -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror)
target_compile_definitions(pusch_mimo_tx_chain PRIVATE
    CUSTOM_ASCEND310P SOC_VERSION="${SOC_VERSION}")
target_link_libraries(pusch_mimo_tx_chain PRIVATE
    host_intf_pub ascendc_kernels_${RUN_MODE}
    tiling_api register platform ascendalog dl)

install(TARGETS pusch_mimo_tx_chain RUNTIME DESTINATION bin)
