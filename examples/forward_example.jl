using SeismicWaves, HDF5

function forward_example()
    ##========================================
    # numerics
    nt = 1500
    dt = 0.0012
    dh = 10.0
    t = collect(Float64, range(0.0; step=dt, length=nt)) # time vector
    ##========================================
    # create a velocity model (gradient from top to bottom)
    nx = 211
    nz = 120
    velmod = zeros(nx, nz)
    for i in 1:nx
        for j in 1:nz
            velmod[i, j] = 2000.0 + 12.0 * (j - 1)
        end
    end
    matprop = VpAcousticCDMaterialProperties(velmod)
    ##========================================
    # shots definition
    nshots = 6
    shots = Vector{Shot{Float64}}()  #Pair{Sources, Receivers}}()
    # sources x-position (in grid points) (different for every shot)
    ixsrc = round.(Int, LinRange(32, nx - 31, nshots))
    for i in 1:nshots
        # sources definition
        nsrc = 1
        possrcs = zeros(1, 2)                # 1 source, 2 dimensions
        possrcs[:, 1] .= (ixsrc[i] - 1) * dh    # x-positions in meters
        possrcs[:, 2] .= 2 * dh                 # y-positions in meters
        # source time functions
        srcstf = zeros(nt, nsrc)
        for s in 1:nsrc
            srcstf[:, s] .= 1000.0 .* rickerstf.(t, 1.20 / 12.0, 12.0)
        end
        srcs = ScalarSources(possrcs, srcstf, 12.0)

        # receivers definition
        nrecs = 20
        # receivers x-positions (in grid points) (same for every shot)
        ixrec = round.(Int, LinRange(30, nx - 29, nrecs))
        posrecs = zeros(nrecs, 2)              # 20 receivers, 2 dimensions
        posrecs[:, 1] .= (ixrec .- 1) .* dh    # x-positions in meters
        posrecs[:, 2] .= 2 * dh                # y-positions in meters
        recs = ScalarReceivers(posrecs, nt)

        # add pair as shot
        push!(shots, Shot(; srcs=srcs, recs=recs)) # srcs => recs)
    end
    ##============================================
    ## Input parameters for acoustic simulation
    parall = :serial
    snapevery = 50
    infoevery = 500
    boundcond = CPMLBoundaryConditionParameters(; halo=20, rcoef=0.0001, freeboundtop=true)
    params = InputParametersAcoustic(nt, dt, (nx, nz), (dh, dh), boundcond)
    ## Show parameters
    dump(params)
    ##===============================================
    ## Compute the seismograms
    snapshots = swforward!(
        params,
        matprop,
        shots;
        parall=parall,
        infoevery=infoevery,
        snapevery=snapevery
    )

    return params, velmod, shots, snapshots
end

## Run the example
params, velmod, shots, snapshots = forward_example()

## Save results
h5open("forward_example.h5", "w") do file
    @write file velmod
    for (i, shot) in enumerate(shots)
        write(file, "seismograms_shot$i", shot.recs.seismograms)
    end
    for (i, snaps) in enumerate(snapshots)
        write(file, "snaps_shot$i", snaps)
    end
end
