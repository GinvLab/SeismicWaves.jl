
#
# MIT License
# Copyright (c) 2019 Andrea Zunino
# 

################################################################
################################################################

"""
A convenience module to facilitate the use of `SeismicWaves` within the framework of Hamiltonian Monte Carlo inversion by employing the package `HMCtomo`. 
"""
module HMCseiswaves

using SeismicWaves.AcousticWaves
using SeismicWaves.ElasticWaves
using LinearAlgebra

export AcouWavProb,ElaWavProb


#################################################################

## create the problem type for traveltime tomography
Base.@kwdef struct AcouWavProb
    inpars::InpParamAcou
    ijsrcs::Vector{Array{Int64,2}}
    ijrecs::Vector{Array{Int64,2}}
    sourcetf::Vector{Array{Float64,2}}
    srcdomfreq::Vector{Float64}
    dobs::Vector{Array{Float64,2}}
    invCovds::Union{Vector{Matrix{Float64}},Vector{Diagonal{Float64}}}
    runparallel::Bool
end

## use  x.T * C^-1 * x  = ||L^-1 * x ||^2 ?

## make the type callable
function (acouprob::AcouWavProb)(vecvel::Vector{Float64},kind::String)

    # reshape vector to 2D array
    vel2d = reshape(vecvel,acouprob.inpars.nx,acouprob.inpars.nz)

    if kind=="nlogpdf"
        #############################################
        ## compute the logdensity value for vecvel ##
        #############################################
        misval = acoumisfitfunc(acouprob.inpars, acouprob.ijsrcs, vel2d, acouprob.ijrecs,
                                acouprob.sourcetf, acouprob.srcdomfreq,
                                acouprob.dobs, acouprob.invCovds,
                                runparallel=acouprob.runparallel)
        return misval        

    elseif kind=="gradnlogpdf"
        #################################################
        ## compute the gradient of the misfit function ##
        #################################################
        grad = gradacoustic2D(acouprob.inpars,acouprob.dobs,acouprob.invCovds,
                              acouprob.ijsrcs,vel2d,acouprob.ijrecs,
                              acouprob.sourcetf,acouprob.srcdomfreq,
                              runparallel=acouprob.runparallel)
        # return flattened gradient
        return  vec(grad)
        
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
