
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

using SeismicWaves

export AcouWavProb,ElaWavProb


#################################################################

## create the problem type for traveltime tomography
Base.@kwdef struct AcouWavProb
    inpars::InpParamAcou
    ijsrcs::Vector{Array{Int64,2}}
    vel::Array{Float64,2}
    ijrecs::Vector{Array{Int64,2}}
    sourcetf::Vector{Array{Float64,2}}
    srcdomfreq::Vector{Float64}
    dobs::Array{Float64,2}
    ## ???
end

## use  x.T * C^-1 * x  = ||L^-1 * x ||^2 ?

## make the type callable
function (acouprob::AcouWavProb)(vecvel::Vector{Float64},kind::String)

    # reshape vector to 2D array
    velnd = reshape(vecvel,eikprob.grd.nx,eikprob.grd.ny)


    if kind=="nlogpdf"
        #############################################
        ## compute the logdensity value for vecvel ##
        #############################################

        

        return misval
        

    elseif kind=="ngradlogpdf"
        #################################################
        ## compute the gradient of the misfit function ##
        #################################################


        # flatten array

        # return flattened gradient
        return  vecgrad
        
    else
        error("Wrong argument 'kind'...")
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
