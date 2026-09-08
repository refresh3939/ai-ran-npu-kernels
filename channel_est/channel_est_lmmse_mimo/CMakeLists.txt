cmake_minimum_required(VERSION 3.16)
project(channel_est_lmmse LANGUAGES C CXX)

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend310P1" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "ASCEND CANN package installation directory")
set(CE_NR "64" CACHE STRING "number of receive antennas")
set(CE_NL "2" CACHE STRING "number of active separated layers (1/2/3/4)")
set(CE_RANK "96" CACHE STRING "frequency LMMSE retained rank")
set(CE_BLOCK_DIM "4" CACHE STRING "number of AI Core blocks")
option(CE_CUBE_TIME_FUSED "Fold time interpolation weights into Cube right operands" OFF)
option(CE_CUBE_TIME_POST_GEMM "Run time interpolation as a post-frequency Cube GEMM" OFF)
if(CE_CUBE_TIME_FUSED AND CE_CUBE_TIME_POST_GEMM)
    message(FATAL_ERROR "CE_CUBE_TIME_FUSED and CE_CUBE_TIME_POST_GEMM are mutually exclusive")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "" FORCE)
endif()

set(KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/channel_est_lmmse_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/channel_est_lmmse_adapters_kernel.cpp)
set(CUSTOM_ASCEND310P_LIST "Ascend310P1" "Ascend310P3")

if("${RUN_MODE}" STREQUAL "cpu")
    include(cmake/cpu_lib.cmake)
elseif("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
    include(cmake/npu_lib.cmake)
endif()

add_executable(ascendc_kernels_bbit
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/channel_est_lmmse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/channel_est_lmmse_tiling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/pusch_mimo_runtime_config.cpp)
target_include_directories(ascendc_kernels_bbit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_compile_options(ascendc_kernels_bbit PRIVATE
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:-g>>
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Werror)
target_compile_definitions(ascendc_kernels_bbit PRIVATE
    $<$<BOOL:$<IN_LIST:${SOC_VERSION},${CUSTOM_ASCEND310P_LIST}>>:CUSTOM_ASCEND310P>
    CE_NR=${CE_NR}
    CE_NL=${CE_NL}
    CE_RANK=${CE_RANK}
    CE_BLOCK_DIM=${CE_BLOCK_DIM}
    $<$<BOOL:${CE_CUBE_TIME_FUSED}>:CE_CUBE_TIME_FUSED=1>
    $<$<BOOL:${CE_CUBE_TIME_POST_GEMM}>:CE_CUBE_TIME_POST_GEMM=1>
    SOC_VERSION="${SOC_VERSION}")
target_link_libraries(ascendc_kernels_bbit PRIVATE
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:ascendcl>>
    ascendc_kernels_${RUN_MODE} tiling_api register platform ascendalog dl)
install(TARGETS ascendc_kernels_bbit RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

enable_testing()
add_executable(channel_est_lmmse_contract_test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/host_contract_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/channel_est_lmmse.cpp)
target_include_directories(channel_est_lmmse_contract_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_compile_definitions(channel_est_lmmse_contract_test PRIVATE
    CE_NR=${CE_NR} CE_NL=${CE_NL} CE_RANK=${CE_RANK} CE_BLOCK_DIM=${CE_BLOCK_DIM})
target_compile_options(channel_est_lmmse_contract_test PRIVATE
    -O2 -std=c++17 -Wall -Wextra -Werror)
add_test(NAME channel_est_lmmse_host_contract COMMAND channel_est_lmmse_contract_test)
install(TARGETS channel_est_lmmse_contract_test RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(channel_est_lmmse_data_utils_test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/data_utils_test.cpp)
target_include_directories(channel_est_lmmse_data_utils_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(channel_est_lmmse_data_utils_test PRIVATE
    -O2 -std=c++17 -Wall -Wextra -Werror)
add_test(NAME channel_est_lmmse_data_utils COMMAND channel_est_lmmse_data_utils_test)
