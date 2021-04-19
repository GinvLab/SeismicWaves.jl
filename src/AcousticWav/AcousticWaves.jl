
##============================
@reexport module AcousticWaves

# using Distributed
## using PyPlot

using LinearAlgebra
using DelimitedFiles
using Distributed

export InpParamAcou
export solveacoustic2D
export gradacoustic2D
export acoumisfitfunc


include("acousticwave_core.jl")
include("acousticwaveprop_serial.jl")
include("../parautils.jl")
include("acousticwaveprop_parallel-source.jl")

include("acousticwaveprop_parallel-shared.jl")
include("acousticwaveprop_parallel-threads.jl")


end ##== module ==============

