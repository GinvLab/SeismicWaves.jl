
using Revise
using SeismicWaves
using LinearAlgebra
using GLMakie
using DSP

###################################################################
using Logging

function exeelanoise(dt, nt, f0, t0)

    ##========================================
    # time stuff
    t = collect(Float64, range(0.0; step=dt, length=nt)) # seconds
    #@show dt,(nt-1)*dt

    ##========================================
    # create a velocity model
    nx = 301 # 211
    nz = 301 # 120
    dh = 4.5 # meters
    lx = (nx - 1) * dh
    lz = (nz - 1) * dh
    @show nx, nz, nx * nz, dh
    #@show (nx-1)*dh, (nz-1)*dh

    vp = 2700.0 .* ones(nx, nz)
    #vp .= 2700.0
    # vp[nx÷2+10:end, :] .= 3100.0
    # for i in 1:nx
    #     for j in 1:nz
    #         vp[i, j] = 2000.0 + dh * (j - 1)
    #     end
    # end
    # @show extrema(vp)
    vs = zeros(nx, nz)

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

    # sources definition
    possrcs = zeros(1, 2)    # 1 source, 2 dimensions
    possrcs[1, :] .= (3lx / 8, lz / 4)
    # source time functions
    srcstf = zeros(nt, 2, 1)
    srcstf[:, 1, 1] .= rickerstf.(t, t0, f0)
    srcstf[:, 2, 1] .= rickerstf.(t, t0, f0)
    srcs = PSDMomentTensorSources(possrcs, srcstf, f0,
        [MomentTensor2D(; Mxx=1.0e16, Mzz=1.0e16, Mxz=0.0)])
    
    # receivers definition
    posrecs = zeros(3, 2)    # 3 receivers, 2 dimensions
    posrecs[1, :] .= ( lx / 4,  lz / 2)
    posrecs[2, :] .= (2lx / 4,  lz / 2)
    posrecs[3, :] .= (3lx / 4,  lz / 2)
    refrecidx = 2
    recs = VectorCrossCorrelationsReceivers(posrecs, nt, [refrecidx]) # ref rec is #2

    shots = [PSDMomentTensorShot(; srcs=srcs, recs=recs)]

    ##============================================
    ## Input parameters for elastic simulation
    infoevery = 50
    freetop = false
    halo = 20
    rcoef = 0.0001
    boundcond = CPMLBoundaryConditionParameters(; halo=halo, rcoef=rcoef, freeboundtop=freetop)
    params = InputParametersElastic(nt, dt, (nx, nz), (dh, dh), boundcond)

    ##===============================================
    # ## compute the cross-correlations
    runparams = RunParameters(parall=:threads,infoevery=infoevery)
    swforward!(params, matprop, shots; runparams=runparams)
    ccs = shots[1].recs.crosscorrelations

    ##===============================================
    ## compute the cross-correlations (dumb way)
    srcs = MomentTensorSources(possrcs, srcstf[:, 1, :], [MomentTensor2D(; Mxx=1.0e16, Mzz=1.0e16, Mxz=0.0)], f0)
    recs = VectorReceivers(posrecs, nt)
    shots = [MomentTensorShot(; srcs=srcs, recs=recs)]
    swforward!(params, matprop, shots; runparams=runparams)

    seismograms = shots[1].recs.seismograms

    # for d in 1:2
    #     for r in 1:3
    #         seismograms[2:end-1, d, r] .= (seismograms[3:end, d, r] .- 2 .* seismograms[2:end-1, d, r] .+ seismograms[1:end-2, d, r]) ./ dt^2
    #     end
    # end

    # for d1 in 1:2
    #     for d2 in 1:2
    #         for r1 in 1:1
    #             for r2 in 1:3
    #                 ccs[2:end-1, d1, d2, r1, r2] .= (ccs[3:end, d1, d2, r1, r2] .- 2 .* ccs[2:end-1, d1, d2, r1, r2] .+ ccs[1:end-2, d1, d2, r1, r2]) ./ dt^2
    #             end
    #         end
    #     end
    # end
    ccs = -circshift(ccs, -t0 ÷ dt)

    @show size(seismograms)
    ccs2 = zeros(2nt+1, 2, 2, 1, 3) # nt, ncomp, ncomp, nrec, nrec
    for d1 in 1:2
        for d2 in 1:2
            for r in 1:3
                ccs2[2:end-1, d1, d2, 1, r] .= xcorr(seismograms[:, d1, refrecidx], seismograms[:, d2, r])
            end
        end
    end

    return ccs, ccs2, possrcs, posrecs, refrecidx, vp, vs, ρ, lx, lz
end

T = 1.0
dt = 0.0005
nt = ceil(Int, T/dt)
f0 = 5.0
t0 = 2.0 / f0
ccs, ccs2, possrcs, posrecs, refrecidx, vp, vs, ρ, lx, lz = exeelanoise(dt, nt, f0, t0)

times = collect(Float64, range(dt; step=dt, length=nt))

cc_ricker = xcorr(rickerstf.(times, t0, f0), rickerstf.(times, t0, f0))

begin
    fig = Figure()
    ax = Axis(fig[1, 1], title="Cross-correlations", xlabel="Time (s)", ylabel="Amplitude")
    ax2 = Axis(fig[2, 1], title="Setup", xlabel="x (m)", ylabel="z (m)")
    times = collect(Float64, range(dt; step=dt, length=nt))
    times_cc = cat(-reverse(times), [0.0], times; dims=1)
    for irec in [1]
        for jrec in 2:2
            lines!(ax, times_cc, ccs[:, 1, 1, irec, jrec] ./ maximum(abs.(ccs[:, 1, 1, irec, jrec])); label="Rec $jrec, Comp zz, fast")
            lines!(ax, times_cc, ccs2[:, 1, 1, irec, jrec] ./ maximum(abs.(ccs2[:, 1, 1, irec, jrec])); label="Rec $jrec, Comp zz, dumb")
        end
        lines!(ax, times_cc, rickerstf.(times_cc, 0.0, f0); label="Ricker stf")
        lines!(ax, times_cc[2:end-1], cc_ricker ./ maximum(abs.(cc_ricker)); label="Ricker cc")
    end
    axislegend(ax, position=:lb, title="Legend")

    hm = heatmap!(ax2, 0..lx, 0..lz, vp; label="Vp (m/s)")
    Colorbar(fig[2, 2], hm; label="Vp (m/s)")
    scatter!(ax2, possrcs[:, 1], possrcs[:, 2]; color=:red, markersize=15, marker=:star5, label="PSD Sources")
    scatter!(ax2, posrecs[:, 1], posrecs[:, 2]; color=:blue, markersize=15, marker=:dtriangle, label="Receivers")
    xlims!(ax2, (0, lx))
    ylims!(ax2, (0, lz))
    ax2.yreversed = true    

    fig
end