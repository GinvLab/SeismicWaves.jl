
using Revise
using SeismicWaves
using LinearAlgebra
using JLD2

###################################################################
using Logging

function exacouprob_wavsim(parall=:serial)

    ##========================================
    # time stuff
    nt = 1500
    dt = 0.0012
    t = collect(Float64, range(0.0; step=dt, length=nt)) # seconds
    #@show dt,(nt-1)*dt

    ##========================================
    # create a velocity model
    nx = 301 #300 #211
    nz = 288 #300 #120
    dh = 10.0 # meters
    @show nx,nz

    #@show (nx-1)*dh, (nz-1)*dh

    velmod = zeros(nx, nz)
    for i in 1:nx
        for j in 1:nz
            velmod[i, j] = 2000.0 + 12.0 * (j - 1)
        end
    end

    # # pad the lateral borders because there will be some PML layers there
    # npad = 34
    # left = repeat(velmod[1:1, :], npad, 1)
    # right = repeat(velmod[end:end, :], npad, 1)
    # velmod = vcat(left, velmod, right)
    # # pad also the bottom...
    # bottom = repeat(velmod[:, end:end], 1, npad)
    # velmod = hcat(velmod, bottom)

    matprop = VpAcousticCDMaterialProperties(velmod)

    ##========================================
    # shots definition
    nshots = 6
    @show nshots
    shots = Vector{Shot}()  #Pair{Sources, Receivers}}()
    # sources x-position (in grid points) (different for every shot)
    ixsrc = round.(Int, LinRange(32, nx - 31, nshots))
    for i in 1:nshots
        # sources definition
        nsrc = 1
        possrcs = zeros(nsrc, 2)    # 1 source, 2 dimensions
        possrcs[:, 1] .= (ixsrc[i] - 1) * dh    # x-positions in meters
        possrcs[:, 2] .= 2 * dh               # y-positions in meters

        # source time functions
        f0 = 12.0
        t0 = 1.20 / f0
        srcstf = zeros(nt, nsrc)
        for s in 1:nsrc
            srcstf[:, s] .= 1000.0 .* rickerstf.(t, t0, f0)
        end
        srcs = ScalarSources(possrcs, srcstf, f0)

        #@show srcs.positions

        # receivers definition
        nrecs = 20
        # receivers x-positions (in grid points) (same for every shot)
        ixrec = round.(Int, LinRange(30, nx - 29, nrecs))
        posrecs = zeros(nrecs, 2)    # 20 receivers, 2 dimensions
        posrecs[:, 1] .= (ixrec .- 1) .* dh    # x-positions in meters
        posrecs[:, 2] .= 2 * dh                # y-positions in meters
        recs = ScalarReceivers(posrecs, nt)

        #@show recs.positions

        # add pair as shot
        push!(shots, Shot(; srcs=srcs, recs=recs)) # srcs => recs)
    end

    ##============================================
    ## Input parameters for acoustic simulation
    snapevery = 50
    infoevery = 500
    boundcond = CPMLBoundaryConditionParameters(; halo=20, rcoef=0.0001, freeboundtop=true)
    params = InputParametersAcoustic(nt, dt, [nx, nz], [dh, dh], boundcond)

    #@show (boundcond.halo-1)*dh
    ##===============================================
    ##parall = :threads
    logger = ConsoleLogger(Error)
    ## compute the seismograms

    println("####>>>>  parall = $parall <<<<####")
    if parall==:threads || parall==:threadpersrc
        @show Threads.nthreads()
    end

    
    println("\nswforward! - NO wavesim")
    for i=1:3
        @time snapshots = swforward!(params,
                                     matprop,
                                     shots;
                                     parall=parall,
                                     infoevery=infoevery,
                                     logger=logger )
    end

    println("\nswforward! - wavesim")        
    wavesim_fwd = build_wavesim(params; parall=parall,
                                gradient=false)
    
    for i=1:3
        @time snapshots = swforward!(wavesim_fwd,
                                     matprop,
                                     shots;
                                     logger=logger )
    end  
    
    #snapevery=snapevery)

    ##===============================================
    ## compute the gradient
    shots_grad = Vector{Shot}()
    for i in 1:nshots
        seis = shots[i].recs.seismograms
        nt = size(seis, 1)
        recs_grad = ScalarReceivers(shots[i].recs.positions, nt; observed=seis,
            invcov=Diagonal(ones(nt)))
        push!(shots_grad, Shot(; srcs=shots[i].srcs, recs=recs_grad))
    end

    newvelmod = matprop.vp .- 0.2
    newvelmod[30:40, 33:44] *= 0.9
    matprop_grad = VpAcousticCDMaterialProperties(newvelmod)

    ##
    check_freq = ceil(Int, sqrt(params.ntimesteps)) #nothing #200 #ceil(Int, sqrt(params.ntimesteps))
    @show check_freq
    grad = nothing

    println("\nswgradient! - NO wavesim")
    for i=1:3
        @time grad = swgradient!(params,
                                 matprop_grad,
                                 shots_grad;
                                 check_freq=check_freq,
                                 parall=parall,
                                 logger=logger )
    end


    println("\nswgradient! - wavesim")
    wavesim_grad = build_wavesim(params; parall=parall,
                                 gradient=true,
                                 check_freq=check_freq)
    
    for i=1:3
        @time grad = swgradient!(wavesim_grad,
                                 matprop_grad,
                                 shots_grad;
                                 logger=logger )
    end

    
    jldsave("fwd_grad_$(parall).jld",shots=shots,grad=grad)

    return params, velmod, shots, grad
end

##################################################################

