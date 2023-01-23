module Acoustic3D_CUDA

using CUDA
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@init_parallel_stencil(CUDA, Float64, 3)

include("shared/Acoustic3D_xPU.jl")

end