module Acoustic1D_CUDA

using CUDA
using ParallelStencil
using ParallelStencil.FiniteDifferences1D
@init_parallel_stencil(CUDA, Float64, 1)

include("shared/Acoustic1D_xPU.jl")

end