module Acoustic1D_Threads

using ParallelStencil
using ParallelStencil.FiniteDifferences1D
@init_parallel_stencil(Threads, Float64, 1)

include("shared/Acoustic1D_xPU.jl")

export forward_onestep!

end