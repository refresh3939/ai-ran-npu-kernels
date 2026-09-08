cmake_minimum_required(VERSION 3.16)
project(mimo_detect_io_pack LANGUAGES C CXX)

enable_testing()

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend310P3" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "Ascend CANN package")
set(IO_PACK_RX_CAPACITY "64" CACHE STRING "detector RX storage capacity")
set(IO_PACK_LAYER_CAPACITY "16" CACHE STRING "detector layer storage capacity")
if(NOT IO_PACK_RX_CAPACITY MATCHES "^(16|32|64)$")
    message(FATAL_ERROR "IO_PACK_RX_CAPACITY must be 16, 32, or 64")
endif()
if(NOT IO_PACK_LAYER_CAPACITY STREQUAL "16")
    message(FATAL_ERROR "Cube detector requires IO_PACK_LAYER_CAPACITY=16")
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
    MIMO_DETECT_IO_NR=${IO_PACK_RX_CAPACITY}
    MIMO_DETECT_IO_NL=${IO_PACK_LAYER_CAPACITY})
if("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
    ascendc_compile_definitions(ascendc_kernels_${RUN_MODE} PRIVATE
        MIMO_DETECT_IO_NR=${IO_PACK_RX_CAPACITY}
        MIMO_DETECT_IO_NL=${IO_PACK_LAYER_CAPACITY})
endif()

add_executable(ascendc_kernels_bbit
    main.cpp
    mimo_detect_io_pack.cpp
    mimo_detect_io_pack_runtime.cpp
    mimo_detect_io_pack_tiling.cpp)
target_include_directories(ascendc_kernels_bbit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(ascendc_kernels_bbit PRIVATE
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:-g>>
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror)
target_compile_definitions(ascendc_kernels_bbit PRIVATE
    $<$<BOOL:$<IN_LIST:${SOC_VERSION},${CUSTOM_ASCEND310P_LIST}>>:CUSTOM_ASCEND310P>
    MIMO_DETECT_IO_NR=${IO_PACK_RX_CAPACITY}
    MIMO_DETECT_IO_NL=${IO_PACK_LAYER_CAPACITY}
    SOC_VERSION="${SOC_VERSION}")
target_link_libraries(ascendc_kernels_bbit PRIVATE
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:ascendcl>>
    ascendc_kernels_${RUN_MODE} tiling_api register platform ascendalog dl)
install(TARGETS ascendc_kernels_bbit RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(mimo_detect_io_pack_contract_test
    tests/test_contract.cpp
    mimo_detect_io_pack.cpp)
target_include_directories(mimo_detect_io_pack_contract_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(mimo_detect_io_pack_contract_test PRIVATE cxx_std_17)
target_compile_options(mimo_detect_io_pack_contract_test PRIVATE
    -O2 -Wall -Wextra -Werror)
target_compile_definitions(mimo_detect_io_pack_contract_test PRIVATE
    MIMO_DETECT_IO_NR=${IO_PACK_RX_CAPACITY}
    MIMO_DETECT_IO_NL=${IO_PACK_LAYER_CAPACITY})
add_test(NAME mimo_detect_io_pack_contract
    COMMAND mimo_detect_io_pack_contract_test)
