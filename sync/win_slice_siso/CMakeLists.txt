cmake_minimum_required(VERSION 3.16)
project(win_slice)

# ── 与 rx_chain / ofdm_demod 同构 ──────────────────────────────
set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend310P3" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "ASCEND CANN package installation directory")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "" FORCE)
endif()

# 设备源 -> ascendc kernel lib(变量名必须是 KERNEL_FILES,npu_lib.cmake 消费它)
set(KERNEL_FILES "")
file(GLOB k "${CMAKE_CURRENT_SOURCE_DIR}/win_slice_kernel.cpp")
list(APPEND KERNEL_FILES ${k})

set(CUSTOM_ASCEND310P_LIST "Ascend310P1" "Ascend310P3")

# cmake/{cpu_lib,npu_lib}.cmake 复制自任一 kernel 目录(与 rx_chain 同样前置)
if("${RUN_MODE}" STREQUAL "cpu")
    include(cmake/cpu_lib.cmake)
elseif("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
    include(cmake/npu_lib.cmake)
endif()

add_executable(win_slice ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

target_include_directories(win_slice PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(win_slice PRIVATE
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall
)

target_compile_definitions(win_slice PRIVATE
    $<$<BOOL:$<IN_LIST:${SOC_VERSION},${CUSTOM_ASCEND310P_LIST}>>:CUSTOM_ASCEND310P>
    SOC_VERSION="${SOC_VERSION}"
)

target_link_libraries(win_slice PRIVATE
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
    ascendc_kernels_${RUN_MODE}
    tiling_api register platform ascendalog dl
)

install(TARGETS win_slice DESTINATION bin)