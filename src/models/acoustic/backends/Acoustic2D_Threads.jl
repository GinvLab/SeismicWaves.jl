module Acoustic2D_Threads

using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(Threads, Float64, 2)

include("shared/Acoustic2D_xPU.jl")

end