module Elastic2D_Iso_CPML_CUDA

using CUDA
using ParallelStencil
using ParallelStencil.FiniteDifferences2D

using SeismicWaves.FiniteDifferencesMacros
using SeismicWaves.FDGeneratedFunctions

@init_parallel_stencil(package = CUDA, ndims = 2, inbounds = false)

include("shared/standard_xPU.jl")
include("shared/freesurface_derivatives.jl")
include("shared/elastic2D_iso_xPU.jl")
include("shared/correlate_gradient_xPU.jl")

end
