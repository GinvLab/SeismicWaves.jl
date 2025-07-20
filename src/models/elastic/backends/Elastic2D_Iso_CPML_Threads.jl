module Elastic2D_Iso_CPML_Threads

using ParallelStencil
using ParallelStencil.FiniteDifferences2D

using SeismicWaves.FiniteDifferencesMacros
using SeismicWaves.FDGeneratedFunctions

@init_parallel_stencil(package = Threads, ndims = 2, inbounds = false)

include("shared/standard_xPU.jl")
include("shared/elastic2D_iso_xPU.jl")
include("shared/correlate_gradient_xPU.jl")
include("shared/smooth_gradient_2D.jl")

end
