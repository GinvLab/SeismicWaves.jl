
#
# MIT License
# Copyright (c) 2023 Andrea Zunino
# 

################################################################
################################################################

"""
A convenience module to facilitate the use of `SeismicWaves` within the framework of Hamiltonian Monte Carlo inversion by employing the package `HMCtomo`.
"""
module HMCseiswaves

using ..SeismicWaves
using LinearAlgebra

export AcouWavProb

#################################################################

## create the problem type for traveltime tomography
Base.@kwdef struct AcouWavProb
    inpars::InputParametersAcoustic{2}
    ijsrcs::Vector{Array{Int64, 2}}
    ijrecs::Vector{Array{Int64, 2}}
    sourcetf::Vector{Array{Float64, 2}}
    srcdomfreq::Vector{Float64}
    dobs::Vector{Array{Float64, 2}}
    invCovds::Vector{<:AbstractMatrix{Float64}}
    parall::Symbol
end

## use  x.T * C^-1 * x  = ||L^-1 * x ||^2 ?

############################################################
## make the type callable
function (acouprob::AcouWavProb)(vecvel::Vector{Float64}, kind::Symbol)
    # numerics
    dh = acouprob.inpars.gridspacing[1]
    nt = acouprob.inpars.ntimesteps

    # reshape vector to 2D array
    vel2d = reshape(vecvel, acouprob.inpars.ns...)
    matprop = VpAcousticCDMaterialProperty(vel2d)

    # build pairs of sources and receivers
    nshots = length(acouprob.ijsrcs)
    shots = Vector{Pair{Sources, Receivers}}()
    for s in 1:nshots
        # sources definition
        possrcs = (acouprob.ijsrcs[s] .- 1) .* dh           # positions in meters
        # source time functions
        f0 = acouprob.srcdomfreq[s]
        srcs = Sources(possrcs, acouprob.sourcetf[s], f0)
        # receivers definition
        posrecs = (acouprob.ijrecs[s] .- 1) .* dh           # positions in meters
        recs = Receivers(posrecs,
            nt;
            observed=acouprob.dobs[s],
            invcov=acouprob.invCovds[s])
        # add pair as shot
        push!(shots, srcs => recs)
    end

    if kind == :nlogpdf
        #############################################
        ## compute the logdensity value for vecvel ##
        #############################################
        misval = swmisfit!(acouprob.inpars, matprop, shots; parall=acouprob.parall)
        return misval

    elseif kind == :gradnlogpdf
        #################################################
        ## compute the gradient of the misfit function ##
        #################################################
        grad = swgradient!(acouprob.inpars,
            matprop,
            shots;
            parall=acouprob.parall,
            check_freq=ceil(Int, sqrt(nt)))
        # return flattened gradient
        return vec(grad)

    elseif kind == :calcforw
        ####################################################
        ## compute calculated data (solve forward problem) ##
        ####################################################
        dcalc = swforward!(acouprob.inpars, matprop, shots; parall=acouprob.parall)
        return dcalc

    else
        error("acouprob::AcouWavProb(): Wrong argument 'kind': $kind...")
    end
end

#################################################################

end # module
#################################################################

####################################
#  A template
#
# struct MyProblem
#     field1::<type>
#     field2::<type>
#     ...
# end
#
# function (myprob::MyProblem)(m::Vector{Float64},kind::String)
#
#     if kind=="logpdf"
#
#         [...]
#         return logpdf  # must return a scalar Float64
#
#     elseif kind=="gradlogpdf"
#
#         [....]
#         return gradlogpdf  # must return an array Vector{Float64}
#
#     else
#         error("Wrong kind...")
#     end
# end
####################################################
