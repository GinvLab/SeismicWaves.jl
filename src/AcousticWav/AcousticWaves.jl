
##============================
module AcousticWaves

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
include("acousticwaveprop_parallel.jl")


end ##== module ==============

