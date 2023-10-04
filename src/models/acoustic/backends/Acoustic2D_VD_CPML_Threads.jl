module Acoustic2D_VD_CPML_Threads

using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(Threads, Float64, 2)

include("shared/standard_xPU.jl")
include("shared/correlate_gradient_xPU.jl")
include("shared/acoustic2D_VD_xPU.jl")
include("shared/smooth_gradient_2D.jl")

end
