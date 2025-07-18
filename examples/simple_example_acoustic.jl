
using Revise
using SeismicWaves
using LinearAlgebra

###################################################################
using Logging

function exacouprob(parall=:serial)

    ##========================================
    # time stuff
    nt = 1500
    dt = 0.0012
    t = collect(Float64, range(0.0; step=dt, length=nt)) # seconds
    #@show dt,(nt-1)*dt

    ##========================================
    # create a velocity model
    nx = 300 #211
    nz = 300 #120
    dh = 10.0 # meters

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
    shots = Vector{ScalarShot{Float64}}()  #Pair{Sources, Receivers}}()
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
        push!(shots, ScalarShot(; srcs=srcs, recs=recs)) # srcs => recs)
    end

    ##============================================
    ## Input parameters for acoustic simulation
    snapevery = 50
    infoevery = 200
    boundcond = CPMLBoundaryConditionParameters(; halo=20, rcoef=0.0001, freeboundtop=true)
    params = InputParametersAcoustic(nt, dt, (nx, nz), (dh, dh), boundcond)

    #@show (boundcond.halo-1)*dh
    ##===============================================
    ##parall = :threads
    logger = ConsoleLogger(Info)
    runparams = RunParameters(parall=:threads,infoevery=infoevery,snapevery=snapevery,logger=logger)

    ## compute the seismograms
    @time snapshots = swforward!(params,
                                 matprop,
                                 shots;
                                 runparams=runparams
                                 )
    seis = copy(shots[1].recs.seismograms)

    @time snapshots = swforward!(params,
                                 matprop,
                                 shots;
                                 runparams=runparams
                                 )
    #snapevery=snapevery)

    seis2 = copy(shots[1].recs.seismograms)

    @assert all(seis .== seis2)

    ##===============================================
    ## compute the gradient
    shots_grad = Vector{ScalarShot{Float64}}()
    for i in 1:nshots
        seis = shots[i].recs.seismograms
        nt = size(seis, 1)
        recs_grad = ScalarReceivers(shots[i].recs.positions, nt; observed=seis,
            invcov=1.0 * I(nt))
        push!(shots_grad, ScalarShot(; srcs=shots[i].srcs, recs=recs_grad))
    end

    newvelmod = matprop.vp .- 0.2
    newvelmod[30:40, 33:44] *= 0.9
    matprop_grad = VpAcousticCDMaterialProperties(newvelmod)

    @time grad = swgradient!(params,
        matprop_grad,
        shots_grad;
        runparams=runparams)

    return params, velmod, shots#, snapshots, grad
end

##################################################################

# debug_logger = ConsoleLogger(stderr, Logging.Debug)
# global_logger(debug_logger)
# error_logger = ConsoleLogger(stderr, Logging.Error)
# global_logger(error_logger)
# info_logger = ConsoleLogger(stderr, Logging.Info)
# global_logger(info_logger)

# par, vel, sh, snaps, grad = exacouprob()

# with_logger(error_logger) do
#     p, v, s, snaps = exacouprob()
# end

# using Plots
# heatmap(snaps[6][:,:,20]'; aspect_ratio=:equal, cmap=:RdBu)
# yaxis!(flip=true)

##################################################################
