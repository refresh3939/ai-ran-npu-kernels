cmake_minimum_required(VERSION 3.16)
project(scramble_mimo LANGUAGES C CXX)

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend310P1" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "Ascend CANN package")
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

add_executable(ascendc_kernels_bbit
    main.cpp
    scramble_mimo.cpp
    scramble_mimo_tiling.cpp)
target_include_directories(ascendc_kernels_bbit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(ascendc_kernels_bbit PRIVATE
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:-g>>
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror)
target_compile_definitions(ascendc_kernels_bbit PRIVATE
    $<$<BOOL:$<IN_LIST:${SOC_VERSION},${CUSTOM_ASCEND310P_LIST}>>:CUSTOM_ASCEND310P>
    SOC_VERSION="${SOC_VERSION}")
target_link_libraries(ascendc_kernels_bbit PRIVATE
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:ascendcl>>
    ascendc_kernels_${RUN_MODE} tiling_api register platform ascendalog dl)
install(TARGETS ascendc_kernels_bbit RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
