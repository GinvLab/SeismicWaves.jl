module Acoustic3D_Threads

using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@init_parallel_stencil(Threads, Float64, 3)

include("shared/Acoustic3D_xPU.jl")

end