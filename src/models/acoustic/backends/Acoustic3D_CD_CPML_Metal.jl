module Acoustic3D_CD_CPML_Metal

using Metal
using ParallelStencil
using ParallelStencil.FiniteDifferences3D

using SeismicWaves.FiniteDifferencesMacros
using SeismicWaves.FDGeneratedFunctions

@init_parallel_stencil(package = Metal, ndims = 3, inbounds = true)

include("shared/standard_xPU.jl")
include("shared/correlate_gradient_xPU.jl")
include("shared/acoustic3D_xPU.jl")
include("shared/smooth_gradient_3D.jl")

end
