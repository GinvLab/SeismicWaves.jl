
module Elastic2D_Iso_CPML_AMDGPU

using AMDGPU
using ParallelStencil
using ParallelStencil.FiniteDifferences2D

using SeismicWaves.FiniteDifferencesMacros
using SeismicWaves.FDGeneratedFunctions

@init_parallel_stencil(package = AMDGPU, ndims = 2, inbounds = true)

include("shared/standard_xPU.jl")
include("shared/freesurface_derivatives_4th_mirror.jl")
include("shared/elastic2D_iso_xPU.jl")
include("shared/correlate_gradient_xPU.jl")

end
