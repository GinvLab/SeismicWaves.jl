module Acoustic1D_CD_CPML_AMDGPU

using AMDGPU
using ParallelStencil
using ParallelStencil.FiniteDifferences1D

using SeismicWaves.FiniteDifferencesMacros
using SeismicWaves.FDGeneratedFunctions

@init_parallel_stencil(package = AMDGPU, ndims = 1, inbound = true)

include("shared/standard_xPU.jl")
include("shared/correlate_gradient_xPU.jl")
include("shared/acoustic1D_xPU.jl")
include("shared/smooth_gradient_1D.jl")

end
