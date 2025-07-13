
using Revise
using SeismicWaves
using LinearAlgebra
using GLMakie

###################################################################
using Logging

function exelaprob()

    ##========================================
    # time stuff
    nt = 3000 #1500
    dt = 0.0008
    t = collect(Float64, range(0.0; step=dt, length=nt)) # seconds
    #@show dt,(nt-1)*dt

    ##========================================
    # create a velocity model
    nx = 380 # 211
    nz = 270 # 120
    dh = 4.5 # meters
    @show nx, nz, nx * nz, dh
    #@show (nx-1)*dh, (nz-1)*dh

    vp = zeros(nx, nz)
    #vp .= 2700.0
    vp[nx÷2+10:end, :] .= 3100.0
    for i in 1:nx
        for j in 1:nz
            vp[i, j] = 2000.0 + dh * (j - 1)
        end
    end
    @show extrema(vp)
    vs = vp ./ sqrt(3)

    @show extrema(vp), extrema(vs)

    ρ = 2100.0 * ones(nx, nz)
    μ = vs .^ 2 .* ρ  # μ = Vs^2⋅ρ 
    λ = (vp .^ 2 .* ρ) .- (2 .* μ)  # λ = vp^2 · ρ - 2μ

    @show extrema(vp .^ 2)
    @show extrema(vp .^ 2 .* ρ)

    @show extrema(λ)
    @show extrema(μ)
    @show extrema(ρ)
    matprop = ElasticIsoMaterialProperties(; λ=λ, μ=μ, ρ=ρ)

    ##========================================
    # shots definition
    nshots = 1
    shots = Vector{MomentTensorShot{Float64, 2, MomentTensor2D{Float64}}}()  #Pair{Sources, Receivers}}()

    for i in 1:nshots
        # sources definition
        nsrc = 1
        # sources x-position (in grid points) (different for every shot)
        if nsrc == 1
            ixsrc = [nx / 2]
        else
            ixsrc = round.(Int, LinRange(30, nx - 31, nsrc))
        end
        possrcs = zeros(nsrc, 2)    # 1 source, 2 dimensions
        for s in 1:nsrc
            possrcs[s, 1] = (ixsrc[i] - 1) * dh .+ 0.124   # x-positions in meters
            possrcs[s, 2] = (nz / 2) * dh .+ 0.124 #(nz/2) * dh              # y-positions in meters
        end

        # source time functions
        f0 = 12.0
        t0 = 1.20 / f0
        srcstf = zeros(nt, nsrc)
        Mxx = zeros(nsrc)
        Mzz = zeros(nsrc)
        Mxz = zeros(nsrc)
        for s in 1:nsrc
            srcstf[:, s] .= rickerstf.(t, t0, f0)
            Mxx[s] = 5e10 #1.5e10  #1.5e6 #e20
            Mzz[s] = 5e10 #2.4e10  #1.5e6 #e20
            Mxz[s] = 0.89e10 #0.89e10 #0.0e6 #e20
        end

        srcs = MomentTensorSources(possrcs, srcstf,
            [MomentTensor2D(; Mxx=Mxx[s], Mzz=Mzz[s], Mxz=Mxz[s]) for s in 1:nsrc],
            f0)
        #srcs = ScalarSources(possrcs, srcstf, f0)

        #@show srcs.positions

        # receivers definition
        nrecs = 10
        # receivers x-positions (in grid points) (same for every shot)
        ixrec = round.(Int, LinRange(40, nx - 40, nrecs))
        posrecs = zeros(nrecs, 2)    # 20 receivers, 2 dimensions
        posrecs[:, 1] .= (ixrec .- 1) .* dh .- 0.324   # x-positions in meters
        posrecs[:, 2] .= 3 * dh # (nz/2) * dh                # y-positions in meters

        ndim = 2
        recs = VectorReceivers(posrecs, nt, ndim)

        #@show recs.positions

        # add pair as shot
        push!(shots, MomentTensorShot(; srcs=srcs, recs=recs)) # srcs => recs)
    end

    @show shots[1].srcs.positions
    @show shots[1].recs.positions

    ##============================================
    ## Input parameters for elastic simulation
    snapevery = 5
    infoevery = 100
    freetop = true
    halo = 20
    rcoef = 0.0001
    @show halo
    @show rcoef
    boundcond = CPMLBoundaryConditionParameters(; halo=halo, rcoef=rcoef, freeboundtop=freetop)
    params = InputParametersElastic(nt, dt, (nx, nz), (dh, dh), boundcond)

    ##===============================================
    ## compute the seismograms
    snapshots = swforward!(params,
        matprop,
        shots;
        parall=:threads,
        infoevery=infoevery,
        snapevery=snapevery)

    # ##===============================================
    # ## compute the gradient
    # shots_grad = Vector{ScalarShot{Float64}}()
    # for i in 1:nshots
    #     seis = shots[i].recs.seismograms
    #     nt = size(seis,1)
    #     recs_grad = ScalarReceivers(shots[i].recs.positions, nt; observed=seis,
    #                                 invcov=Diagonal(ones(nt)))
    #     push!(shots_grad, MomentTensorShot(; srcs=shots[i].srcs, recs=recs_grad))
    # end

    # newvelmod = matprop.vp .- 0.2
    # newvelmod[30:40,33:44] *= 0.9
    # matprop_grad = ElasticIsoMaterialProperties(newvelmod)

    # grad = swgradient!(params,
    #                    matprop_grad,
    #                    shots_grad;
    #                    parall=:threads )

    return params, matprop, shots, snapshots#, grad
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

    vp = sqrt.((matprop.λ + 2 .* matprop.μ) ./ matprop.ρ)

    lsrec = 6:10

    fig = Figure(; size=(1000, 1200))

    ax1 = Axis(fig[1, 1]; title="Ux")
    ax2 = Axis(fig[2, 1]; title="Uz")
    ax3 = Axis(fig[3, 1]; title="Source time function")
    ax4 = Axis(fig[4, 1]; xlabel="x [m]", ylabel="z [m]")

    for r in lsrec
        lines!(ax1, shots[1].recs.seismograms[:, 1, r]; label="Ux #$r")
    end
    axislegend(ax1)

    for r in lsrec
        lines!(ax2, shots[1].recs.seismograms[:, 2, r]; label="Uz #$r")
    end
    axislegend(ax2)

    lines!(ax3, shots[1].srcs.tf[:, 1])

    #poly!(ax4,Rect(rect...),color=:green,alpha=0.3)
    hm = heatmap!(ax4, xgrd, ygrd, vp; colormap=:Reds) #,alpha=0.7)
    Colorbar(fig[4, 2], hm; label="Vp")

    scatter!(ax4, xrec, yrec; marker=:dtriangle, label="Receivers", markersize=15)
    scatter!(ax4, xsrc, ysrc; label="Sources", markersize=15)
    axislegend(ax4)
    for r in lsrec
        text!(ax4, xrec[r], yrec[r]; text="#$r")
    end
    ax4.yreversed = true

    save("ux_uz_vp_example.png", fig)
    return fig
end

function snapanimate(par, matprop, shots, snapsh; scalamp=0.01, snapevery=5)
    xgrd = [par.gridspacing[1] * (i - 1) for i in 1:par.gridsize[1]]
    ygrd = [par.gridspacing[2] * (i - 1) for i in 1:par.gridsize[2]]
    xrec = shots[1].recs.positions[:, 1]
    yrec = shots[1].recs.positions[:, 2]
    xsrc = shots[1].srcs.positions[:, 1]
    ysrc = shots[1].srcs.positions[:, 2]

    vxsnap = [snapsh[1][kk]["ucur"].value[1] for kk in sort(keys(snapsh[1]))]
    vzsnap = [snapsh[1][kk]["ucur"].value[2] for kk in sort(keys(snapsh[1]))]
    @show size(vxsnap), size(vzsnap)

    curvx = Observable(vxsnap[1])
    curvz = Observable(vzsnap[1])
    @show typeof(curvx), size(curvx[])

    vp = sqrt.((matprop.λ + 2 .* matprop.μ) ./ matprop.ρ)

    dx = par.gridspacing[1]
    dz = par.gridspacing[2]
    nx = par.gridsize[1]
    nz = par.gridsize[2]
    halo = par.boundcond.halo
    rcoef = par.boundcond.rcoef
    freetop = par.boundcond.freeboundtop

    if freetop
        rectpml = [dx * (halo - 1), 0.0,
            dx * (nx - 2 * halo - 1), dz * (nz - halo - 1)]
    else
        rectpml = [dx * (halo - 1), dz * (halo - 1),
            dx * (nx - 2 * halo - 1), dz * (nz - 2 * halo - 1)]
    end

    ##=====================================
    fig = Figure(; size=(800, 1500))

    nframes = length(vxsnap)

    cmapwavefield = :vik #:cyclic_grey_15_85_c0_n256_s25 #:balance

    ax1 = Axis(fig[1, 1]; aspect=DataAspect(),
        xlabel="x [m]", ylabel="z [m]", title="Ux, clip at $scalamp of max amplitude, iteration 0 of $(snapevery*nframes)")
    #poly!(ax4,Rect(rect...),color=:green,alpha=0.3)
    extx = extrema.([vxsnap[i] for i in 1:length(vxsnap)])
    extx = map(p -> max(abs(p[1]), abs(p[2])), extx)
    vmax = max(extx...)
    vminmax = scalamp .* (-vmax, vmax)
    hm = heatmap!(ax1, xgrd, ygrd, curvx; colormap=cmapwavefield,
        colorrange=vminmax) #,alpha=0.7)
    Colorbar(fig[1, 2], hm; label="x displ.")

    lines!(ax1, Rect(rectpml...); color=:green)
    scatter!(ax1, xrec, yrec; marker=:dtriangle, label="Receivers", markersize=15)
    scatter!(ax1, xsrc, ysrc; label="Sources", markersize=15)
    # axislegend(ax1)
    ax1.yreversed = true

    ax2 = Axis(fig[2, 1]; aspect=DataAspect(),
        xlabel="x [m]", ylabel="z [m]", title="Uz, clip at $scalamp of max amplitude, iteration 0 of $(snapevery*nframes)")
    #poly!(ax4,Rect(rect...),color=:green,alpha=0.3)
    extx = extrema.([vzsnap[i] for i in 1:length(vzsnap)])
    extx = map(p -> max(abs(p[1]), abs(p[2])), extx)
    vmax = max(extx...)
    vminmax = scalamp .* (-vmax, vmax)
    hm = heatmap!(ax2, xgrd, ygrd, curvz; colormap=cmapwavefield,
        colorrange=vminmax) #,alpha=0.7)
    Colorbar(fig[2, 2], hm; label="z displ.")

    lines!(ax2, Rect(rectpml...); color=:green)
    scatter!(ax2, xrec, yrec; marker=:dtriangle, label="Receivers", markersize=15)
    scatter!(ax2, xsrc, ysrc; label="Sources", markersize=15)
    # axislegend(ax2)
    ax2.yreversed = true

    # ax3 = Axis(fig[3, 1]; aspect=DataAspect(),
    #     xlabel="x [m]", ylabel="z [m]")
    # hm = heatmap!(ax3, xgrd, ygrd, vp; colormap=:Reds) #,alpha=0.7)
    # Colorbar(fig[3, 2], hm; label="Vp")

    # scatter!(ax3, xrec, yrec; marker=:dtriangle, label="Receivers", markersize=15)
    # scatter!(ax3, xsrc, ysrc; label="Sources", markersize=15)
    # axislegend(ax3)
    # ax3.yreversed = true

    ##
    display(fig)
    save("first_frame.png", fig)
    ##=====================================

    function updatefunction(curax1, curax2, vxsnap, vzsnap, it)
        cvx = vxsnap[it]
        cvz = vzsnap[it]
        curax1.title = "Ux, clip at $scalamp of max amplitude, iteration $(snapevery*it) of $(snapevery*nframes)"
        curax2.title = "Uz, clip at $scalamp of max amplitude, iteration $(snapevery*it) of $(snapevery*nframes)"
        return cvx, cvz
    end

    fps = 30

    # live plot
    # for j in 1:1
    #     for it=1:nframes
    #         curvx[],curvz[] = updatefunction(ax1,ax2,vxsnap,vzsnap,it)
    #         sleep(1/fps)
    #     end
    # end

    ##
    record(fig, "snapshots_halo_$(halo)_rcoef_$(rcoef).mp4", 1:nframes; framerate=fps) do it
        curvx[], curvz[] = updatefunction(ax1, ax2, vxsnap, vzsnap, it)
        # yield() -> not required with record
    end
end

##################################################################
# debug_logger = ConsoleLogger(stderr, Logging.Debug)
# global_logger(debug_logger)
# error_logger = ConsoleLogger(stderr, Logging.Error)
# global_logger(error_logger)
info_logger = ConsoleLogger(stderr, Logging.Info)
global_logger(info_logger)

par, matprop, shots, snapsh = exelaprob()

snapanimate(par, matprop, shots, snapsh; scalamp=0.02)

fig = plotstuff(par, matprop, shots)

# with_logger(error_logger) do
#     p, v, s, snaps = exacouprob()
# end

# using Plots
# heatmap(snaps[6][:,:,20]'; aspect_ratio=:equal, cmap=:RdBu)
# yaxis!(flip=true)

##################################################################
