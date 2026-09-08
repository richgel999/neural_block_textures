# CUDA backend of ntc: a static library linked into the ntc executable when NTC_CUDA is ON.
# Needs CUDA >= 12.8 for sm_120 (RTX 5090); on this machine that is CUDA 13.1, selected with
# the Visual Studio generator via the toolset (-T cuda=13.1) or the "cuda" preset.
enable_language(CUDA)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
  message(FATAL_ERROR "CUDA ${CMAKE_CUDA_COMPILER_VERSION} cannot target sm_120; configure with the 'cuda' preset (CUDA 13.1)")
endif()
find_package(CUDAToolkit 12.8 REQUIRED)

add_library(ntc_cuda STATIC ntc_cuda.cu ntc_cuda.h ntc_noise.h)
target_include_directories(ntc_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(ntc_cuda PROPERTIES
  CUDA_ARCHITECTURES "120"
  CUDA_STANDARD 17
  CUDA_SEPARABLE_COMPILATION OFF)
# fp32 with FMA contraction (nvcc's default), deliberately no --use_fast_math: the kernels
# are expected to agree with the CPU reference to float rounding.
target_compile_options(ntc_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_link_libraries(ntc_cuda PUBLIC CUDA::cudart_static)
