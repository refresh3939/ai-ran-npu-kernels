cmake_minimum_required(VERSION 3.16)
project(pusch_mimo_loopback_host_check LANGUAGES CXX)

add_executable(pusch_mimo_loopback_check loopback.cpp)
target_compile_features(pusch_mimo_loopback_check PRIVATE cxx_std_17)
target_compile_options(pusch_mimo_loopback_check PRIVATE -O2 -Wall -Wextra -Werror)
install(TARGETS pusch_mimo_loopback_check RUNTIME DESTINATION bin)
