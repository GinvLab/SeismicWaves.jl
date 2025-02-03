module Elastic2D_Iso_CPML_AMDGPU

using AMDGPU
using ParallelStencil
using ParallelStencil.FiniteDifferences2D

using SeismicWaves.FiniteDifferencesMacros

@init_parallel_stencil(package = AMDGPU, ndims = 2, inbounds = false)

include("shared/standard_xPU.jl")
include("shared/elastic2D_iso_xPU.jl")
include("shared/correlate_gradient_xPU.jl")

end
