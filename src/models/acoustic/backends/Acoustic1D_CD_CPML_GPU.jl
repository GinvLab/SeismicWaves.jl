module Acoustic1D_CD_CPML_GPU

using CUDA
using ParallelStencil
using ParallelStencil.FiniteDifferences1D
@init_parallel_stencil(CUDA, Float64, 1)

include("shared/standard_xPU.jl")
include("shared/correlate_gradient_xPU.jl")
include("shared/acoustic1D_xPU.jl")

end
