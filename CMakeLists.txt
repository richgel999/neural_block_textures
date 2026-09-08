cmake_minimum_required(VERSION 3.20)
project(ntc LANGUAGES CXX)

option(NTC_CUDA "Build the CUDA backend (--cuda); needs CUDA >= 12.8 for sm_120" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP)

add_executable(ntc main.cpp)
target_include_directories(ntc PRIVATE ${CMAKE_SOURCE_DIR}/cuda)   # ntc_noise.h is plain C++, used by --rng hash
if(OpenMP_CXX_FOUND)
  target_link_libraries(ntc PRIVATE OpenMP::OpenMP_CXX)
endif()

# Host flags, scoped to C++ so they never reach nvcc.
if(MSVC)
  target_compile_options(ntc PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W3 /fp:fast /arch:AVX2> $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CONFIG:Debug>>>:/O2>)
else()
  target_compile_options(ntc PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -ffast-math -march=native> $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CONFIG:Debug>>>:-O3>)
endif()

# Standalone SSE4.1 CPU decoder. No /arch flag on MSVC: SSE2 baseline codegen, SSE4.1 only through
# explicit intrinsics, so no FMA contraction is possible; --exact / --fast are run-time choices.
add_executable(ntc_decode ntc_decode.cpp)
if(MSVC)
  target_compile_options(ntc_decode PRIVATE /W3 /fp:fast $<$<NOT:$<CONFIG:Debug>>:/O2>)
else()
  target_compile_options(ntc_decode PRIVATE -Wall -msse4.1 -ffast-math $<$<NOT:$<CONFIG:Debug>>:-O3>)
  find_package(Threads REQUIRED)
  target_link_libraries(ntc_decode PRIVATE Threads::Threads)
endif()

# Debug builds get a _d suffix (ntc_d / ntc_decode_d) so a Debug binary is never mistaken for a Release one;
# the /O2 (-O3) above is scoped to the non-Debug configurations so a Debug build really is unoptimized.
# Both programs print "DEBUG build" as their first line when DEBUG or _DEBUG is defined.
set_target_properties(ntc ntc_decode PROPERTIES DEBUG_POSTFIX "_d")

if(NTC_CUDA)
  add_subdirectory(cuda)
  target_link_libraries(ntc PRIVATE ntc_cuda)
  target_compile_definitions(ntc PRIVATE NTC_CUDA=1)
endif()
