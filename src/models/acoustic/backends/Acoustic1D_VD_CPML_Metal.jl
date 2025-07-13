module Acoustic1D_VD_CPML_Metal

using Metal
using ParallelStencil
using ParallelStencil.FiniteDifferences1D

using SeismicWaves.FiniteDifferencesMacros
using SeismicWaves.FDGeneratedFunctions

@init_parallel_stencil(package = Metal, ndims = 1, inbounds = true)

include("shared/standard_xPU.jl")
include("shared/correlate_gradient_xPU.jl")
include("shared/acoustic1D_VD_xPU.jl")
include("shared/smooth_gradient_1D.jl")

end
