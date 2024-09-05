module Elastic2D_Iso_CPML_CUDA

using CUDA
using ParallelStencil
using ParallelStencil.FiniteDifferences2D

@init_parallel_stencil(package = CUDA, ndims = 2, inbounds = false)

include("shared/standard_xPU.jl")
include("shared/fourth_order_FiniteDifferences2D.jl")
include("shared/elastic2D_iso_xPU.jl")
include("shared/correlate_gradient_xPU.jl")

end
