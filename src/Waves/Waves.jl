"""
Module for generalized wave propagation solvers.
"""
module Waves

include("models/wavemodel.jl")

include("traits/equation.jl")
include("traits/kernel.jl")
include("traits/boundarycondition.jl")
include("traits/shooting.jl")
include("traits/snappable.jl")

include("sources.jl")
include("receivers.jl")
include("check.jl")
include("precompute.jl")
include("forward.jl")
include("init_shot.jl")
include("solve.jl")

include("models/acoustic/check.jl")
include("models/acoustic/precompute.jl")
include("models/acoustic/models.jl")
include("models/acoustic/forward.jl")
export IsotropicAcousticSerialReflectiveWaveModel1D

export Sources, Receivers
export solve!

include("models/acoustic/backends/acoustic1D.jl")

include("utils.jl")
export rickersource1D

end