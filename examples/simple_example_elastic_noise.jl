
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

    # sources definition
    possrcs = zeros(1, 2)    # 1 source, 2 dimensions
    possrcs[1, :] .= (3lx / 8, lz / 4)
    # source time functions
    srcstf = zeros(nt, 2, 1)
    srcstf[:, 1, 1] .= rickerstf.(t, t0, f0)
    srcstf[:, 2, 1] .= rickerstf.(t, t0, f0)
    srcs = PSDMomentTensorSources(possrcs, srcstf, f0,
        [MomentTensor2D(; Mxx=1.0, Mzz=1.0, Mxz=0.0)])
    
    # receivers definition
    posrecs = zeros(3, 2)    # 3 receivers, 2 dimensions
    posrecs[1, :] .= ( lx / 4,  lz / 2)
    posrecs[2, :] .= (2lx / 4,  lz / 2)
    posrecs[3, :] .= (3lx / 4,  lz / 2)
    recs = VectorCrossCorrelationsReceivers(posrecs, nt, [2], [1]) # ref rec is #2, only compute #1 component

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
    srcs = MomentTensorSources(possrcs, srcstf[:, 1, :], [MomentTensor2D(; Mxx=1.0, Mzz=1.0, Mxz=0.0)], f0)
    recs = VectorReceivers(posrecs, nt)
    shots = [MomentTensorShot(; srcs=srcs, recs=recs)]
    swforward!(params, matprop, shots; runparams=runparams)

    seismograms = shots[1].recs.seismograms

    @show size(seismograms)
    cc_xx_21 = xcorr(seismograms[:, 1, 2], seismograms[:, 1, 1])
    cc_xx_22 = xcorr(seismograms[:, 1, 2], seismograms[:, 1, 2])
    cc_xx_23 = xcorr(seismograms[:, 1, 2], seismograms[:, 1, 3])
    @show size(cc_xx_21), size(cc_xx_22), size(cc_xx_23)
    ccs2 = zeros(2nt+1, 1, 1, 1, 3) # nt, ncomp, ncomp, nrec, nrec
    ccs2[2:end-1, 1, 1, 1, 1] .= cc_xx_21
    ccs2[2:end-1, 1, 1, 1, 2] .= cc_xx_22
    ccs2[2:end-1, 1, 1, 1, 3] .= cc_xx_23

    return ccs, ccs2
end

T = 0.75
dt = 0.0005
nt = ceil(Int, T/dt)
f0 = 10.0
t0 = 2.0 / f0
ccs, ccs2 = exeelanoise(dt, nt, f0, t0)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], title="Cross-correlations", xlabel="Time (s)", ylabel="Amplitude")
    ax2 = Axis(fig[2, 1], title="Cross-correlations (dumb way)", xlabel="Time (s)", ylabel="Amplitude")
    times = collect(Float64, range(dt; step=dt, length=nt))
    times_cc = cat(-reverse(times), [0.0], times; dims=1)
    for irec in [1]
        for jrec in 1:3
            for d in 1:1
                lines!(ax, times_cc, ccs[:, d, d, irec, jrec]; label="Rec $jrec, Comp $d-$d")
                lines!(ax2, times_cc, ccs2[:, d, d, irec, jrec]; label="Rec $jrec, Comp $d-$d")
            end
        end
    end
    vlines!(ax, t0, color=:red, label="t0", linestyle=:dash)
    axislegend(ax, position=:lb, title="Legend")
    axislegend(ax2, position=:lb, title="Legend")
    fig
end