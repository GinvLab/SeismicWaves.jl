
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
using Logging

export AcouWavCDProb

#wavesim = nothing

#################################################################

## create the problem type for traveltime tomography
struct AcouWavCDProb
    #wavesim::WaveSimulation
    inpars::InputParametersAcoustic
    shots::Vector{<:Shot} #invCovds::Vector{<:AbstractMatrix{Float64}}
    parall::Symbol
    #firsttime::Base.RefValue{Bool}

    function AcouWavCDProb(inpars::InputParametersAcoustic,
        shots::Vector{<:Shot},
        parall::Symbol)

        #     nt = inpars.ntimesteps
        #     check_freq = ceil(Int, sqrt(nt))
        #     wavesim = build_wavesim(inpars; gradient=true,check_freq=check_freq,
        #                             snapevery=nothing,infoevery=nothing)
        return new(inpars, shots, parall) #,Ref(true))
    end
end

## use  x.T * C^-1 * x  = ||L^-1 * x ||^2 ?

############################################################
## make the type callable
function (acouprob::AcouWavCDProb)(vecvel::Vector{Float64}, kind::Symbol)
    logger = Logging.ConsoleLogger(Error)
    #logger = Logging.NullLogger()

    # if acouprob.firsttime[]
    #     acouprob.firsttime[] = false
    #     nt = acouprob.inpars.ntimesteps
    #     check_freq = ceil(Int, sqrt(nt))
    #     global wavesim = build_wavesim(acouprob.inpars; parall=acouprob.parall,
    #                                    gradient=true,check_freq=check_freq,
    #                                    snapevery=nothing,infoevery=nothing)
    # end
    # @show HMCseiswaves.firsttime

    # # numerics
    # dh = acouprob.inpars.gridspacing[1]
    # nt = acouprob.inpars.ntimesteps

    # reshape vector to 2D array
    velNd = reshape(vecvel, acouprob.inpars.gridsize...)
    matprop = VpAcousticCDMaterialProperties(velNd)

    @assert length(acouprob.shots[1].recs.observed) != 0

    # @show BLAS.get_num_threads()
    # @show acouprob.parall
    # @show Threads.nthreads()

    if kind == :nlogpdf
        #############################################
        ## compute the logdensity value for vecvel ##
        #############################################
        misval = swmisfit!(acouprob.inpars, matprop, acouprob.shots;
            parall=acouprob.parall, logger=logger)
        # misval = swmisfit!(wavesim, matprop, acouprob.shots,
        #                    logger=logger)
        return misval

    elseif kind == :gradnlogpdf
        #################################################
        ## compute the gradient of the misfit function ##
        #################################################
        grad = swgradient!(acouprob.inpars,
            matprop,
            acouprob.shots;
            parall=acouprob.parall,
            ## if next line is commented: no checkpointing
            check_freq=ceil(Int, sqrt(acouprob.inpars.ntimesteps)),
            logger=logger)
        # @time grad = swgradient!(wavesim,
        #                          matprop,
        #                          acouprob.shots,
        #                          logger=logger)

        # return flattened gradient
        return vec(grad)

    elseif kind == :calcforw
        ####################################################
        ## compute calculated data (solve forward problem) ##
        ####################################################
        dcalc = swforward!(acouprob.inpars, matprop, acouprob.shots;
            parall=acouprob.parall, logger=logger)
        # dcalc = swforward!(wavesim, matprop, acouprob.shots,
        #                    logger=logger)
        return dcalc

    else
        error("acouprob::AcouWavProbCD(): Wrong argument 'kind': $kind...")
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
