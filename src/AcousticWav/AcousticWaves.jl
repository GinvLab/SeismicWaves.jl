
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
export para_solveacoustic2D,para_gradacoustic2D

include("acousticwaveprop.jl")
include("../parautils.jl")
include("acousticwaveprop_parallel.jl")


end ##== module ==============

