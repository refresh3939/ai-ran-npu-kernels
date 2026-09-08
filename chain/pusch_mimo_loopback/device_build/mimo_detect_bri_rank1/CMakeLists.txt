cmake_minimum_required(VERSION 3.16)
project(pusch_mimo_detect_bri_rank1 LANGUAGES C CXX)
set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend310P1" CACHE STRING "target")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE STRING "CANN")
set(CLEAN_KERNEL_ROOT "/home/refresh/AI-RAN-NPU-clean/kernels" CACHE PATH "categorized kernel root")
set(ACTIVE_LAYERS "1" CACHE STRING "active Rank1..4 layer count")
set(DETECTOR_RX_CAPACITY "64" CACHE STRING "selected detector RX storage capacity")
set(DETECTOR_LAYER_CAPACITY "16" CACHE STRING "selected detector layer storage capacity")
set(DETECTOR_BRI_BLOCK "8" CACHE STRING "BRI preconditioner block size")
if(NOT ACTIVE_LAYERS MATCHES "^[1-4]$")
  message(FATAL_ERROR "ACTIVE_LAYERS must be 1, 2, 3, or 4")
endif()
if(NOT DETECTOR_RX_CAPACITY MATCHES "^(16|32|64)$" OR
   NOT DETECTOR_LAYER_CAPACITY STREQUAL "16")
  message(FATAL_ERROR "detector storage must be 16/32/64 Rx by 16 layers")
endif()
if(NOT DETECTOR_BRI_BLOCK MATCHES "^(1|2|4|8)$")
  message(FATAL_ERROR "DETECTOR_BRI_BLOCK must be 1, 2, 4, or 8")
endif()
set(op "${CLEAN_KERNEL_ROOT}/mimo/mimo_detect_bri_batch")
foreach(required mimo_detect_bri_kernel.cpp mimo_detect_bri.h mimo_detect_bri_host_contract.cpp mimo_detect_bri_tiling.cpp main.cpp)
  if(NOT EXISTS "${op}/${required}")
    message(FATAL_ERROR "missing categorized BRI source ${op}/${required}")
  endif()
endforeach()
set(KERNEL_FILES "${op}/mimo_detect_bri_kernel.cpp")
set(N_SC_PAD_VAL 1664)
set(CUSTOM_ASCEND310P_LIST Ascend310P1 Ascend310P3)
include_directories("${op}")
include("${op}/cmake/npu_lib.cmake")
# The promoted header documents but omits these fixed production schedule
# constants.  Inject them at the build boundary without modifying clean.
ascendc_compile_definitions(ascendc_kernels_npu PRIVATE
  CUSTOM_ASCEND310P HAVE_WORKSPACE HAVE_TILING N_SC_PAD_VAL=1664
  MIMO_M=${DETECTOR_RX_CAPACITY} MIMO_K=${DETECTOR_LAYER_CAPACITY}
  MIMO_KR=${ACTIVE_LAYERS} BRI_B=${DETECTOR_BRI_BLOCK}
  GROUP_BATCH=8 RE_WINDOW=5824 N_WINDOW_PER_CORE=1)
ascendc_compile_options(ascendc_kernels_npu PRIVATE -forward-options-to-host-compiler
  -DMIMO_M=${DETECTOR_RX_CAPACITY} -DMIMO_K=${DETECTOR_LAYER_CAPACITY}
  -DMIMO_KR=${ACTIVE_LAYERS} -DBRI_B=${DETECTOR_BRI_BLOCK}
  -DGROUP_BATCH=8 -DRE_WINDOW=5824 -DN_WINDOW_PER_CORE=1)
add_executable(ascendc_kernels_bbit "${op}/main.cpp"
  "${op}/mimo_detect_bri_host_contract.cpp" "${op}/mimo_detect_bri_tiling.cpp")
target_include_directories(ascendc_kernels_bbit PRIVATE "${op}")
target_compile_options(ascendc_kernels_bbit PRIVATE -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Werror)
target_compile_definitions(ascendc_kernels_bbit PRIVATE CUSTOM_ASCEND310P SOC_VERSION="${SOC_VERSION}"
  N_SC_PAD_VAL=1664 MIMO_M=${DETECTOR_RX_CAPACITY}
  MIMO_K=${DETECTOR_LAYER_CAPACITY} MIMO_KR=${ACTIVE_LAYERS}
  BRI_B=${DETECTOR_BRI_BLOCK} MIMO_IO_NR_VALUE=${DETECTOR_RX_CAPACITY}
  MIMO_IO_LAYER_VALUE=${DETECTOR_LAYER_CAPACITY}
  GROUP_BATCH=8 RE_WINDOW=5824 N_WINDOW_PER_CORE=1)
target_link_libraries(ascendc_kernels_bbit PRIVATE host_intf_pub ascendc_kernels_npu
  tiling_api register platform ascendalog dl)
install(TARGETS ascendc_kernels_bbit RUNTIME DESTINATION bin)
