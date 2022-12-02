"""
Module for generalized wave propagation solvers.
"""
module Waves

include("traits/model.jl")
include("traits/equation.jl")
include("traits/kernel.jl")
include("traits/boundarycondition.jl")
include("traits/shooting.jl")

include("sources.jl")
include("receivers.jl")
include("forward.jl")
include("solve.jl")

export Sources, Receivers
export solve!

end