module Acoustic2D_CD_CPML_GPU

using CUDA
using ParallelStencil
using ParallelStencil.FiniteDifferences2D

@init_parallel_stencil(package=CUDA, ndims=2, inbounds=true)

include("shared/standard_xPU.jl")
include("shared/correlate_gradient_xPU.jl")
include("shared/acoustic2D_xPU.jl")
include("shared/smooth_gradient_2D.jl")

end
