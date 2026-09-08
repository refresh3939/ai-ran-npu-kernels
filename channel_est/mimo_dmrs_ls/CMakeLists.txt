cmake_minimum_required(VERSION 3.16)
project(mimo_dmrs_ls LANGUAGES C CXX)

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend310P1" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "Ascend CANN package")
set(DMRS_LS_NR "64" CACHE STRING "execution RX storage capacity")
set(DMRS_LS_BLOCK_DIM "4" CACHE STRING "DMRS-LS AI Core count")
if(NOT DMRS_LS_NR MATCHES "^(16|32|64)$")
    message(FATAL_ERROR "DMRS_LS_NR must be 16, 32, or 64")
endif()
math(EXPR DMRS_LS_EXPECTED_BLOCKS "${DMRS_LS_NR} / 16")
if(NOT DMRS_LS_BLOCK_DIM STREQUAL "${DMRS_LS_EXPECTED_BLOCKS}")
    message(FATAL_ERROR "DMRS_LS_BLOCK_DIM must equal DMRS_LS_NR/16")
endif()
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "" FORCE)
endif()

file(GLOB KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*_kernel.cpp)
set(CUSTOM_ASCEND310P_LIST "Ascend310P1" "Ascend310P3")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
if("${RUN_MODE}" STREQUAL "cpu")
    include(cmake/cpu_lib.cmake)
elseif("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
    include(cmake/npu_lib.cmake)
endif()
target_compile_definitions(ascendc_kernels_${RUN_MODE} PRIVATE
    MIMO_DMRS_LS_NR=${DMRS_LS_NR}
    MIMO_DMRS_LS_BLOCK_DIM=${DMRS_LS_BLOCK_DIM})
if("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
    ascendc_compile_definitions(ascendc_kernels_${RUN_MODE} PRIVATE
        MIMO_DMRS_LS_NR=${DMRS_LS_NR}
        MIMO_DMRS_LS_BLOCK_DIM=${DMRS_LS_BLOCK_DIM})
endif()

add_executable(ascendc_kernels_bbit main.cpp mimo_dmrs_ls.cpp)
target_include_directories(ascendc_kernels_bbit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(ascendc_kernels_bbit PRIVATE
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:-g>>
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror)
target_compile_definitions(ascendc_kernels_bbit PRIVATE
    $<$<BOOL:$<IN_LIST:${SOC_VERSION},${CUSTOM_ASCEND310P_LIST}>>:CUSTOM_ASCEND310P>
    MIMO_DMRS_LS_NR=${DMRS_LS_NR}
    MIMO_DMRS_LS_BLOCK_DIM=${DMRS_LS_BLOCK_DIM}
    SOC_VERSION="${SOC_VERSION}")
target_link_libraries(ascendc_kernels_bbit PRIVATE
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:ascendcl>>
    ascendc_kernels_${RUN_MODE} tiling_api register platform ascendalog dl)
install(TARGETS ascendc_kernels_bbit RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(mimo_dmrs_ls_contract_test scripts/test_contract.cpp mimo_dmrs_ls.cpp)
target_include_directories(mimo_dmrs_ls_contract_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(mimo_dmrs_ls_contract_test PRIVATE
    -O2 -std=c++17 -Wall -Wextra -Werror)
target_compile_definitions(mimo_dmrs_ls_contract_test PRIVATE
    MIMO_DMRS_LS_NR=${DMRS_LS_NR}
    MIMO_DMRS_LS_BLOCK_DIM=${DMRS_LS_BLOCK_DIM})
