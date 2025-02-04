module Acoustic2D_CD_CPML_Threads

using ParallelStencil
using ParallelStencil.FiniteDifferences2D

@init_parallel_stencil(package = Threads, ndims = 2, inbounds = true)

include("shared/standard_xPU.jl")
include("shared/correlate_gradient_xPU.jl")
include("shared/acoustic2D_xPU.jl")
include("shared/smooth_gradient_2D.jl")

end
