module Acoustic2D_CUDA

using CUDA
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(CUDA, Float64, 2)

include("shared/Acoustic2D_xPU.jl")

end