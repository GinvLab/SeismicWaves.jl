module Acoustic1D_CD_CPML_CUDA

using CUDA
using ParallelStencil
using ParallelStencil.FiniteDifferences1D

using SeismicWaves.FiniteDifferencesMacros

@init_parallel_stencil(package = CUDA, ndims = 1, inbound = true)

include("shared/standard_xPU.jl")
include("shared/correlate_gradient_xPU.jl")
include("shared/acoustic1D_xPU.jl")
include("shared/smooth_gradient_1D.jl")

end
