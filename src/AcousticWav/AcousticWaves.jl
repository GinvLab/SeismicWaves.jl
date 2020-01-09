
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


include("acousticwaveprop.jl")
include("../parautils.jl")
include("acousticwaveprop_parallel.jl")


##------------------------------------------------------------
"""
  Solver for 2D acoustic wave equation (parameters: velocity only). 
"""
function solveacoustic2D(inpar::InpParamAcou,ijsrcs::Array{Array{Int64,2},1},
                          vel::Array{Float64,2}, ijrecs::Array{Array{Int64,2},1},
                          sourcetf::Array{Array{Float64,2},1}, srcdomfreq::Array{Float64,1};
                          runparallel::Bool=false)

    if runparallel
        output = solveacoustic2D_parallel(inpar,ijsrcs,vel,ijrecs,sourcetf,srcdomfreq)

        if inpar.savesnapshot==true
            receiv,psave = output[1],output[2]
            return receiv,psave
        end

    else
        output = solveacoustic2D_serial(inpar,ijsrcs,vel,ijrecs,sourcetf,srcdomfreq)

        if inpar.savesnapshot==true
            receiv,psave = output[1],output[2]
            return receiv,psave
        end
    end

    return output
end

##------------------------------------------------------------

"""
  Solver for computing the gradient of the misfit function for the acoustic 
   wave equation using the adjoint state method.
"""
function gradacoustic2D(inpar::InpParamAcou, obsrecv::Array{Array{Float64,2},1},
                        invCovds::Union{Vector{Matrix{Float64}},Vector{Diagonal{Float64}}}, 
                        ijsrcs::Array{Array{Int64,2},1},
                        vel::Array{Float64,2}, ijrecs::Array{Array{Int64,2},1},
                        sourcetf::Array{Array{Float64,2},1}, srcdomfreq::Array{Float64,1};
                        runparallel::Bool=false)

    if runparallel
        grad = gradacoustic2D_parallel(inpar,obsrecv,invCovds,ijsrcs,vel,ijrecs,sourcetf,srcdomfreq)
    else
        grad = gradacoustic2D_serial(inpar,obsrecv,invCovds,ijsrcs,vel,ijrecs,sourcetf,srcdomfreq)
    end

    return grad
end

##------------------------------------------------------------

end ##== module ==============

