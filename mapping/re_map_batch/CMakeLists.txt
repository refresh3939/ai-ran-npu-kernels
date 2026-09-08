cmake_minimum_required(VERSION 3.16)
project(re_map_batch LANGUAGES C CXX)

set(RUN_MODE "npu" CACHE STRING "sim/npu")
set(SOC_VERSION "Ascend310P3" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "Ascend CANN package")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "" FORCE)
endif()

# Reuse the verified SISO implementation verbatim. Batching belongs solely to
# the host adapter, as required by the v1 chain contract.
set(KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../re_map_siso/re_map_kernel.cpp)
set(CUSTOM_ASCEND310P_LIST "Ascend310P1" "Ascend310P3")
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../re_map_siso)
if("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
    include(${CMAKE_CURRENT_SOURCE_DIR}/../re_map_siso/cmake/npu_lib.cmake)
else()
    message(FATAL_ERROR "re_map_batch runtime adapter supports RUN_MODE=sim|npu")
endif()

add_executable(ascendc_kernels_bbit main.cpp re_map_batch.cpp)
target_include_directories(ascendc_kernels_bbit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../re_map_siso
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common)
target_compile_options(ascendc_kernels_bbit PRIVATE
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror)
target_compile_definitions(ascendc_kernels_bbit PRIVATE
    $<$<BOOL:$<IN_LIST:${SOC_VERSION},${CUSTOM_ASCEND310P_LIST}>>:CUSTOM_ASCEND310P>
    SOC_VERSION="${SOC_VERSION}")
target_link_libraries(ascendc_kernels_bbit PRIVATE
    host_intf_pub ascendc_kernels_${RUN_MODE}
    tiling_api register platform ascendalog dl)
install(TARGETS ascendc_kernels_bbit RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

enable_testing()
add_executable(re_map_batch_contract_test
    tests/test_contract.cpp
    re_map_batch.cpp)
target_include_directories(re_map_batch_contract_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common)
target_compile_definitions(re_map_batch_contract_test PRIVATE ASCENDC_CPU_DEBUG)
target_compile_options(re_map_batch_contract_test PRIVATE
    -O2 -UNDEBUG -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror)
add_test(NAME re_map_batch_contract COMMAND re_map_batch_contract_test)
