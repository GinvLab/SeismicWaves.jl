module Acoustic1D_CD_CPML_Threads

using ParallelStencil
using ParallelStencil.FiniteDifferences1D
@init_parallel_stencil(package=Threads, ndims=1, inbounds=true)

include("shared/standard_xPU.jl")
include("shared/correlate_gradient_xPU.jl")
include("shared/acoustic1D_xPU.jl")
include("shared/smooth_gradient_1D.jl")

end
