
using Revise
using SeismicWaves
using LinearAlgebra
using Logging
using GLMakie



function exacouprob()

    ##========================================
    # time stuff
    nt = 1500
    dt = 0.001
    t = collect(Float64, range(0.0; step=dt, length=nt)) # seconds
    #@show dt,(nt-1)*dt

    ##========================================
    # create a velocity model
    nx = 300 
    nz = 280 
    dh = 8.0 # meters

    velmod = zeros(nx, nz)
    for i in 1:nx
        for j in 1:nz
            velmod[i, j] = 2000.0 + 12.0 * (j - 1)
        end
    end

    matprop = VpAcousticCDMaterialProperties(velmod)

    ##========================================
    # shots definition
    nshots = 3
    shots = Vector{ScalarShot{Float64}}()  
    # sources x-position (in grid points) (different for every shot)
    ixsrc = round.(Int, LinRange(32, nx - 31, nshots))
    for i in 1:nshots
        # sources definition
        nsrc = 1
        possrcs = zeros(nsrc, 2)               # 1 source, 2 dimensions
        possrcs[:, 1] .= (ixsrc[i] - 1) * dh   # x-positions in meters
        possrcs[:, 2] .= (nz-40) * dh               # y-positions in meters

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
        nrecs = 10
        # receivers x-positions (in grid points) (same for every shot)
        ixrec = round.(Int, LinRange(30, nx - 29, nrecs))
        posrecs = zeros(nrecs, 2)              # 20 receivers, 2 dimensions
        posrecs[:, 1] .= (ixrec .- 1) .* dh    # x-positions in meters
        posrecs[:, 2] .= 2 * dh                # y-positions in meters
        recs = ScalarReceivers(posrecs, nt)
        #@show recs.positions

        # add pair as shot
        push!(shots, ScalarShot(; srcs=srcs, recs=recs)) 
    end

    ##============================================
    ## Input parameters for acoustic simulation
    snapevery = 50
    infoevery = 200
    boundcond = CPMLBoundaryConditionParameters(; halo=20, rcoef=0.0001, freeboundtop=true)
    params = InputParametersAcoustic(nt, dt, (nx, nz), (dh, dh), boundcond)

    ##===============================================
    runparams = RunParameters(parall=:threads,
                              infoevery=infoevery,
                              snapevery=snapevery,
                              logger = ConsoleLogger(Info) )

    ## compute the seismograms
    snapshots = swforward!(params,
                          matprop,
                          shots;
                          runparams=runparams
                          )

    ##===============================================
    ## compute the gradient
    misfit = Vector{SeismicWaves.L2Misfit{Float64}}()
    for i in 1:nshots
        seis = shots[i].recs.seismograms
        nt = size(seis,1)
        invcov = Diagonal(ones(nt))
        push!(misfit,SeismicWaves.L2Misfit(observed=seis,invcov=invcov))
    end

    # modify input velocity model
    newvelmod = matprop.vp .- 0.2
    newvelmod[30:40, 33:44] *= 0.9
    matprop_grad = VpAcousticCDMaterialProperties(newvelmod)

    gradparams = GradParameters(mute_radius_src=5,
                                mute_radius_rec=2,
                                compute_misfit=true)
    
    grad, misfitval = swgradient!(params,
                                  matprop_grad,
                                  shots,
                                  misfit;
                                  runparams=runparams,
                                  gradparams=gradparams)

    return params, matprop, shots, snapshots, grad
end

##################################################################


function plotstuff(par, matprop, shots)
    rect = [0.0, 0.0, par.gridspacing[1] * (par.gridsize[1] - 1), par.gridspacing[2] * (par.gridsize[2] - 1)]

    xgrd = [par.gridspacing[1] * (i - 1) for i in 1:par.gridsize[1]]
    ygrd = [par.gridspacing[2] * (i - 1) for i in 1:par.gridsize[2]]

    xrec = shots[1].recs.positions[:, 1]
    yrec = shots[1].recs.positions[:, 2]
    xsrc = shots[1].srcs.positions[:, 1]
    ysrc = shots[1].srcs.positions[:, 2]

    vp = matprop.vp

    lsrec = 1:2:10

    fig = Figure(; size=(800, 1200))

    ax1 = Axis(fig[1, 1]; title="Pressure")
    ax3 = Axis(fig[2, 1]; title="Source time function")
    ax4 = Axis(fig[3, 1]; xlabel="x [m]", ylabel="z [m]")

    for r in lsrec
        lines!(ax1, shots[1].recs.seismograms[:, r]; label="Pressure #$r")
    end
    axislegend(ax1)

    lines!(ax3, shots[1].srcs.tf[:, 1])

    #poly!(ax4,Rect(rect...),color=:green,alpha=0.3)
    hm = heatmap!(ax4, xgrd, ygrd, vp; colormap=:Reds) #,alpha=0.7)
    Colorbar(fig[3, 2], hm; label="Vp")

    scatter!(ax4, xrec, yrec; marker=:dtriangle, label="Receivers", markersize=15)
    scatter!(ax4, xsrc, ysrc; label="Sources", markersize=15)
    axislegend(ax4)
    for r in lsrec
        text!(ax4, xrec[r], yrec[r]; text="#$r")
    end
    ax4.yreversed = true

    save("acou_example.png", fig)
    return fig
end


##################################################################

par, matprop, shots, snapsh, grad = exacouprob()

fig = plotstuff(par, matprop, shots)
display(fig)


##################################################################
