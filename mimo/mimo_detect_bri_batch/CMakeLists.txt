cmake_minimum_required(VERSION 3.16)
project(Ascend_c)

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend310P3" CACHE STRING "system on chip type")
set(N_SC_PAD_VAL "1664" CACHE STRING "N_SC_PAD debug override")
set(DETECTOR_RX_CAPACITY "64" CACHE STRING "detector RX storage capacity")
set(DETECTOR_LAYER_CAPACITY "16" CACHE STRING "detector physical layer capacity")
set(ACTIVE_LAYERS "16" CACHE STRING "active detector layers")
set(DETECTOR_BRI_BLOCK "8" CACHE STRING "BRI preconditioner block size")
set(DETECTOR_PACK "0" CACHE STRING "RE packing mode")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "ASCEND CANN package installation directory")
if(NOT DETECTOR_RX_CAPACITY MATCHES "^(16|32|64)$" OR
   NOT DETECTOR_LAYER_CAPACITY STREQUAL "16")
    message(FATAL_ERROR "detector storage must be 16/32/64 Rx by 16 layers")
endif()
if(NOT ACTIVE_LAYERS MATCHES "^([1-9]|1[0-6])$")
    message(FATAL_ERROR "ACTIVE_LAYERS must be in [1,16]")
endif()
if(NOT DETECTOR_BRI_BLOCK MATCHES "^(1|2|4|8)$")
    message(FATAL_ERROR "DETECTOR_BRI_BLOCK must be 1, 2, 4, or 8")
endif()
if(NOT DETECTOR_PACK STREQUAL "0")
    message(FATAL_ERROR "production standalone path requires DETECTOR_PACK=0")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "" FORCE)
endif()

file(GLOB KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/mimo_detect_bri_kernel.cpp)

set(CUSTOM_ASCEND310P_LIST "Ascend310P1" "Ascend310P3")

if("${RUN_MODE}" STREQUAL "cpu")
    include(cmake/cpu_lib.cmake)
elseif("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
    include(cmake/npu_lib.cmake)
endif()
target_compile_definitions(ascendc_kernels_${RUN_MODE} PRIVATE
    MIMO_M=${DETECTOR_RX_CAPACITY}
    MIMO_K=${DETECTOR_LAYER_CAPACITY}
    MIMO_KR=${ACTIVE_LAYERS}
    BRI_B=${DETECTOR_BRI_BLOCK}
    AIRAN_PACK=${DETECTOR_PACK}
    GROUP_BATCH=8 RE_WINDOW=5824 N_WINDOW_PER_CORE=1)
if("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
    ascendc_compile_definitions(ascendc_kernels_${RUN_MODE} PRIVATE
        CUSTOM_ASCEND310P HAVE_WORKSPACE HAVE_TILING N_SC_PAD_VAL=${N_SC_PAD_VAL}
        MIMO_M=${DETECTOR_RX_CAPACITY}
        MIMO_K=${DETECTOR_LAYER_CAPACITY}
        MIMO_KR=${ACTIVE_LAYERS}
        BRI_B=${DETECTOR_BRI_BLOCK}
        AIRAN_PACK=${DETECTOR_PACK}
        GROUP_BATCH=8 RE_WINDOW=5824 N_WINDOW_PER_CORE=1)
    ascendc_compile_options(ascendc_kernels_${RUN_MODE} PRIVATE
        -forward-options-to-host-compiler
        -DMIMO_M=${DETECTOR_RX_CAPACITY}
        -DMIMO_K=${DETECTOR_LAYER_CAPACITY}
        -DMIMO_KR=${ACTIVE_LAYERS}
        -DBRI_B=${DETECTOR_BRI_BLOCK}
        -DAIRAN_PACK=${DETECTOR_PACK}
        -DGROUP_BATCH=8 -DRE_WINDOW=5824 -DN_WINDOW_PER_CORE=1)
endif()

add_executable(ascendc_kernels_bbit
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mimo_detect_bri_host_contract.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mimo_detect_bri_tiling.cpp
)

target_include_directories(ascendc_kernels_bbit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
)

target_compile_options(ascendc_kernels_bbit PRIVATE
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:-g>>
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Werror
)

target_compile_definitions(ascendc_kernels_bbit PRIVATE
    $<$<BOOL:$<IN_LIST:${SOC_VERSION},${CUSTOM_ASCEND310P_LIST}>>:CUSTOM_ASCEND310P>
    SOC_VERSION="${SOC_VERSION}"
    N_SC_PAD_VAL=${N_SC_PAD_VAL}
    MIMO_M=${DETECTOR_RX_CAPACITY}
    MIMO_K=${DETECTOR_LAYER_CAPACITY}
    MIMO_KR=${ACTIVE_LAYERS}
    BRI_B=${DETECTOR_BRI_BLOCK}
    AIRAN_PACK=${DETECTOR_PACK}
    MIMO_IO_NR_VALUE=${DETECTOR_RX_CAPACITY}
    MIMO_IO_LAYER_VALUE=${DETECTOR_LAYER_CAPACITY}
    BRI_GROUPED_NR=${DETECTOR_RX_CAPACITY}
    BRI_GROUPED_NL=${DETECTOR_LAYER_CAPACITY}
    GROUP_BATCH=8 RE_WINDOW=5824 N_WINDOW_PER_CORE=1
)

target_link_libraries(ascendc_kernels_bbit PRIVATE
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:ascendcl>>
    ascendc_kernels_${RUN_MODE}
    tiling_api register platform ascendalog dl
)

install(TARGETS ascendc_kernels_bbit
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
