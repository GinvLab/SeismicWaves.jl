using SeismicWaves

###################################################################

function exacouprob()

    ##========================================
    # time stuff
    nt = 1500 
    dt = 0.0012
    t = collect(Float64,range(0.0,step=dt,length=nt)) # seconds
    #@show dt,(nt-1)*dt

    ##========================================
    # create a velocity model
    nx = 211
    nz = 120
    dh = 10.0 # meters

    velmod = zeros(nx,nz)
    for i=1:nx
        for j=1:nz
            velmod[i,j] = 2000.0 + 12.0*(j-1)
        end
    end

    # pad the lateral borders because there will be some PML layers there
    npad = 34
    left = repeat(velmod[1:1,:],npad,1)
    right = repeat(velmod[end:end,:],npad,1)
    velmod = vcat(left,velmod,right)
    # pad also the bottom...
    bottom = repeat(velmod[:,end:end],1,npad)
    velmod = hcat(velmod,bottom)

    ##========================================
    # shots definition
    nshots = 6
    shots = Vector{Pair{Sources, Receivers}}()
    # sources x-position (in grid points) (different for every shot)
    ixsrc = round.(Int, LinRange(32, nx-31, nshots))
    for i=1:nshots
        # sources definition
        nsrc = 1
        possrcs = zeros(nsrc,2)    # 1 source, 2 dimensions
        possrcs[:,1] .= (ixsrc[i]-1) * dh    # x-positions in meters
        possrcs[:,2] .= 2 * dh               # y-positions in meters
        # source time functions
        f0 = 12.0
        t0 = 1.20 / f0
        srcstf = zeros(nt,nsrc)
        for s in 1:nsrc
            srcstf[:,s] .= 1000.0 .* rickersource1D.(t, t0, f0)
        end
        srcs = Sources(possrcs, srcstf, f0)

        # receivers definition
        nrecs = 20
        # receivers x-positions (in grid points) (same for every shot)
        ixrec = round.(Int,LinRange(30, nx-29, nrecs))
        posrecs = zeros(nrecs,2)    # 20 receivers, 2 dimensions
        posrecs[:,1] .= (ixrec .- 1) .* dh    # x-positions in meters
        posrecs[:,2] .= 2 * dh                # y-positions in meters
        recs = Receivers(posrecs, nt)

        # add pair as shot
        push!(shots, srcs => recs)
    end

    ##============================================
    ## Input parameters for acoustic simulation
    snapevery = 50
    infoevery = 500
    boundcond = CPML_BC(halo=20, rcoef=0.0001, freeboundtop=true)
    params = InputParametersAcoustic(
        nt, dt, [nx, nz], [dh, dh], boundcond
    )

    ##===============================================
    ## compute the seismograms
    snapshots = swforward!(params, velmod, shots; use_GPU=false, snapevery=snapevery, infoevery=infoevery)

    return params, velmod, collect(map(s -> s.second.seismograms, shots)), snapshots
end

# p, v, s, snaps = exacouprob()

using Plots
heatmap(snaps[6][:,:,20]'; aspect_ratio=:equal, cmap=:RdBu)
yaxis!(flip=true)


##################################################################

