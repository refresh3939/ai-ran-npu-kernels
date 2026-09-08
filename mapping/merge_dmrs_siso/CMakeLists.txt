# merge_dmrs — host-only operator (TX DMRS comb-2 insertion). No Ascend kernel;
# links libascendcl for aclrtMemcpy (D2H/H2D scatter). Simpler than sibling
# kernel operators (no RUN_MODE/SOC_VERSION cmake).
cmake_minimum_required(VERSION 3.16)
project(merge_dmrs CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ASCEND_HOME_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
    else()
        set(ASCEND_HOME_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    endif()
endif()

add_executable(merge_dmrs_bbit main.cpp merge_dmrs.cpp)
target_include_directories(merge_dmrs_bbit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(merge_dmrs_bbit PRIVATE
    ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(merge_dmrs_bbit ascendcl)
