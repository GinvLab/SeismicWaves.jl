using SeismicWaves, Logging, CairoMakie, ColorSchemes
using LinearAlgebra, Serialization

function setup1D_CD()
    # Set up the parameters for the simulation
    nt = 1000
    c0 = 1500.0
    c0max = 2000.0
    dx = 0.5
    cfl = 0.8
    dt = dx / c0max * cfl
    halo, rcoef = 0, 0.0001
    nx = 1001
    lx = (nx - 1) * dx
    gridx = LinRange(0.0, lx, nx)
    # Set up velocity model
    vmod_inner = LinRange(c0, c0max, nx - 2*halo)
    vmod = zeros(nx)
    vmod[halo+1:nx-halo] .= vmod_inner
    vmod[1:halo] .= vmod[halo+1]
    vmod[nx-halo+1:end] .= vmod[nx-halo]
    matprop = VpAcousticCDMaterialProperties(vmod)
    # Set up the geometry
    f0 = 100.0
    t0 = 2.0 / f0
    times = collect(Float64, range(0.0; step=dt, length=nt))
    srctf = rickerstf.(times, t0, f0)
    srcspos = zeros(1, 1)
    srcspos[1, 1] = 0.3 * lx
    srcstf = zeros(nt, 1)
    srcstf[:, 1] .= srctf
    recspos = zeros(2, 1)
    recspos[1, 1] = 0.7 * lx
    recspos[2, 1] = 0.8 * lx
    observed = zeros(nt, 2)
    shots = [ScalarShot(
        srcs=ScalarSources(srcspos, srcstf, f0),
        recs=ScalarReceivers(recspos, nt; observed=observed, invcov=1.0*I(nt))
    )]
    # Set up the simulation
    boundcond = CPMLBoundaryConditionParameters(; halo=halo, rcoef=rcoef, freeboundtop=false)
    params = InputParametersAcoustic(nt, dt, (nx,), (dx,), boundcond)
    return params, matprop, times, gridx, shots, srcstf
end

function setup2D_CD()
    # Set up the parameters for the simulation
    nt = 700
    c0 = 1500.0
    c0max = 2000.0
    dh = 0.5
    cfl = 0.8
    dt = dh / c0max * cfl / sqrt(2)
    halo, rcoef = 0, 0.0001
    nx, ny = 401, 301
    lx, ly = (nx - 1) * dh, (ny - 1) * dh
    gridx = LinRange(0.0, lx, nx)
    gridy = LinRange(0.0, ly, ny)
    # Set up velocity model
    vmod_inner_1D = LinRange(c0, c0max, ny - halo)
    vmod = zeros(nx, ny)
    for i in 1:nx
        vmod[i, 1:ny-halo] .= vmod_inner_1D
    end
    vmod[:, ny-halo+1:end] .= vmod[:, ny-halo]
    matprop = VpAcousticCDMaterialProperties(vmod)
    # Set up the geometry
    f0 = 100.0
    t0 = 2.0 / f0
    times = collect(Float64, range(0.0; step=dt, length=nt))
    srctf = rickerstf.(times, t0, f0)
    srcspos = zeros(1, 2)
    srcspos[1, :] .= (0.3 * lx, ly / 2) 
    srcstf = zeros(nt, 1)
    srcstf[:, 1] .= srctf
    recspos = zeros(1, 2)
    recspos[1, :] .= (0.7 * lx, ly / 2)
    observed = zeros(nt, 1)
    shots = [ScalarShot(
        srcs=ScalarSources(srcspos, srcstf, f0),
        recs=ScalarReceivers(recspos, nt; observed=observed, invcov=1.0*I(nt))
    )]
    # Set up the simulation
    boundcond = CPMLBoundaryConditionParameters(; halo=halo, rcoef=rcoef, freeboundtop=true)
    params = InputParametersAcoustic(nt, dt, (nx,ny), (dh,dh), boundcond)
    return params, matprop, times, gridx, gridy, shots, srcstf
end

function setup2D_VD()
    # Set up the parameters for the simulation
    nt = 700
    c0 = 1500.0
    c0max = 2000.0
    dh = 0.5
    cfl = 0.8
    dt = dh / c0max * cfl / sqrt(2) * 6/7
    halo, rcoef = 0, 0.0001
    nx, ny = 401, 301
    lx, ly = (nx - 1) * dh, (ny - 1) * dh
    gridx = LinRange(0.0, lx, nx)
    gridy = LinRange(0.0, ly, ny)
    # Set up velocity model
    vmod_inner_1D = LinRange(c0, c0max, ny - halo)
    vmod = zeros(nx, ny)
    for i in 1:nx
        vmod[i, 1:ny-halo] .= vmod_inner_1D
    end
    vmod[:, ny-halo+1:end] .= vmod[:, ny-halo]
    rhomod = vmod .* 0.5
    matprop = VpRhoAcousticVDMaterialProperties(vmod, rhomod)
    # Set up the geometry
    f0 = 100.0
    t0 = 2.0 / f0
    times = collect(Float64, range(0.0; step=dt, length=nt))
    srctf = rickerstf.(times, t0, f0) ./ 1e3
    srcspos = zeros(1, 2)
    srcspos[1, :] .= (0.3 * lx, ly / 2) 
    srcstf = zeros(nt, 1)
    srcstf[:, 1] .= srctf
    recspos = zeros(1, 2)
    recspos[1, :] .= (0.7 * lx, ly / 2)
    observed = zeros(nt, 1)
    shots = [ScalarShot(
        srcs=ScalarSources(srcspos, srcstf, f0),
        recs=ScalarReceivers(recspos, nt; observed=observed, invcov=1.0*I(nt))
    )]
    # Set up the simulation
    boundcond = CPMLBoundaryConditionParameters(; halo=halo, rcoef=rcoef, freeboundtop=true)
    params = InputParametersAcoustic(nt, dt, (nx,ny), (dh,dh), boundcond)
    return params, matprop, times, gridx, gridy, shots, srcstf
end

function setup2D_isoela()
    # Set up the parameters for the simulation
    nt = 1500
    vp0 = 1500.0
    vp0max = 1501.0
    dh = 0.5
    cfl = 0.8
    dt = dh / vp0max * cfl / sqrt(2) * 6/7
    halo, rcoef = 20, 0.0001
    nx, ny = 401, 251
    lx, ly = (nx - 1) * dh, (ny - 1) * dh
    gridx = LinRange(0.0, lx, nx)
    gridy = LinRange(0.0, ly, ny)
    # Set up velocity model
    vmod_inner_1D = LinRange(vp0, vp0max, ny - 2*halo)
    vmod = zeros(nx, ny)
    for i in 1:nx
        vmod[i, halo+1:ny-halo] .= vmod_inner_1D
    end
    vmod[:, 1:halo] .= vmod[:, halo+1]
    vmod[:, ny-halo+1:end] .= vmod[:, ny-halo]
    vpmod = copy(vmod)
    vsmod = vpmod ./ sqrt(3)
    rhomod = vpmod .* 0.5
    μmod = vsmod .^ 2 .* rhomod
    λmod = (vpmod .^ 2 .* rhomod) .- (2 .* μmod)
    matprop = ElasticIsoMaterialProperties(; λ=λmod, μ=μmod, ρ=rhomod)
    # Set up the geometry
    f0 = 50.0
    t0 = 1.5 / f0
    times = collect(Float64, range(0.0; step=dt, length=nt))
    srctf = rickerstf.(times, t0, f0) * 1e10
    srcspos = zeros(1, 2)
    srcspos[1, :] .= (0.45 * lx, ly / 2) 
    srcstf = zeros(nt, 1)
    srcstf[:, 1] .= srctf
    momten = MomentTensor2D(; Mxx=0.78, Mzz=0.65, Mxz=0.34)
    recspos = zeros(1, 2)
    recspos[1, :] .= (0.55 * lx, ly / 2)
    observed = zeros(nt, 2, 1)
    shots = [MomentTensorShot(
        srcs=MomentTensorSources(srcspos, srcstf, [momten], f0),
        recs=VectorReceivers(recspos, nt; observed=observed, invcov=1.0*I(nt))
    )]
    # Set up the simulation
    boundcond = CPMLBoundaryConditionParameters(; halo=halo, rcoef=rcoef, freeboundtop=false)
    params = InputParametersElastic(nt, dt, (nx,ny), (dh,dh), boundcond)
    return params, matprop, times, gridx, gridy, shots, srcstf, momten
end

function check_acoustic_CD(params, matprop, shots; computefd=false)
    # Build wavesim
    wavesim = build_wavesim(params, matprop; gradient=true, smooth_radius=0)
    # Compute gradients
    println("Computing adjoint ∂ψ/∂m and ψ(m)")
    grad, misfit = swgradient!(wavesim, matprop, shots; compute_misfit=true)

    eps = minimum(matprop.vp) / 1e6
    # Compute FD gradient
    if computefd
        println("Computing FD ∂ψ/∂m")
        fdgrad = zeros(size(matprop.vp)...)
        dm = eps
        for i in eachindex(matprop.vp)
            matprop.vp[i] += dm
            println("Computing FD gradient for index $i / $(length(matprop.vp))")
            with_logger(ConsoleLogger(Logging.Error)) do
                fdgrad[i] = (swmisfit!(wavesim, matprop, shots;) - misfit) / dm
            end
            matprop.vp[i] -= dm
        end
        serialize("fdgrad.jls", fdgrad)
    else
        if isfile("fdgrad.jls")
            println("Loading FD gradient from file")
            fdgrad = deserialize("fdgrad.jls")
        else
            println("FD gradient file not found. Set computefd=true to compute it.")
            fdgrad = zeros(size(matprop.vp)...)
        end
    end
    # Compute dot product test
    dm = rand(size(matprop.vp)...)
    dm = dm / norm(dm)
    matprop.vp .+= eps .* dm
    println("Computing ψ(m + ϵΔm) where ϵ = $eps, norm(Δm) = $(norm(dm))")
    misfit_dm = with_logger(ConsoleLogger(Logging.Error)) do
        swmisfit!(wavesim, matprop, shots;)
    end
    lhs = dot(grad["vp"], dm)
    rhs = (misfit_dm - misfit) / eps
    println("⟨∂ψ/∂m, Δm⟩ = $lhs")
    println("(ψ(m + ϵΔm⟩) - ψ(m)) / ϵ = $rhs")
    println("Absolute error: $(lhs - rhs)")
    println("Relative error = $(abs(lhs - rhs) / abs(lhs))")

    return grad, fdgrad
end

function check_acoustic_VD(params, matprop, shots; computefd=false)
    # Build wavesim
    wavesim = build_wavesim(params, matprop; gradient=true, smooth_radius=0)
    # Compute gradients
    println("Computing adjoint ∂ψ/∂m and ψ(m)")
    grad, misfit = swgradient!(wavesim, matprop, shots; compute_misfit=true)

    eps_vp = minimum(matprop.vp) / 1e6
    eps_rho = minimum(matprop.rho) / 1e6
    # Compute FD gradient
    if computefd
        println("Computing FD ∂ψ/∂m with respect to vp")
        fdgrad = zeros(size(matprop.vp)...)
        dm = eps_vp
        for i in eachindex(matprop.vp)
            matprop.vp[i] += dm
            println("Computing FD gradient for index $i / $(length(matprop.vp))")
            with_logger(ConsoleLogger(Logging.Error)) do
                fdgrad[i] = (swmisfit!(wavesim, matprop, shots;) - misfit) / dm
            end
            matprop.vp[i] -= dm
        end
        serialize("fdgrad_vp.jls", fdgrad)

        println("Computing FD ∂ψ/∂m with respect to rho")
        fdgrad = zeros(size(matprop.rho)...)
        dm = eps_rho
        for i in eachindex(matprop.rho)
            matprop.rho[i] += dm
            println("Computing FD gradient for index $i / $(length(matprop.vp))")
            with_logger(ConsoleLogger(Logging.Error)) do
                fdgrad[i] = (swmisfit!(wavesim, matprop, shots;) - misfit) / dm
            end
            matprop.rho[i] -= dm
        end
        serialize("fdgrad_rho.jls", fdgrad)
    else
        if isfile("fdgrad_vp.jls") && isfile("fdgrad_rho.jls")
            println("Loading FD gradient from file")
            fdgrad_vp = deserialize("fdgrad_vp.jls")
            fdgrad_rho = deserialize("fdgrad_rho.jls")
        else
            println("FD gradient file not found. Set computefd=true to compute it.")
            fdgrad_vp = zeros(size(matprop.vp)...)
            fdgrad_rho = zeros(size(matprop.rho)...)
        end
    end
    # Compute dot product test
    dm_vp = rand(size(matprop.vp)...)
    dm_vp = dm_vp ./ norm(dm_vp)
    dm_rho = rand(size(matprop.rho)...)
    dm_rho = dm_rho ./ norm(dm_rho)

    matprop.vp .+= eps_vp .* dm_vp
    println("Computing ψ(vp + ϵΔvp) where ϵ = $(eps_vp), norm(Δvp) = $(norm(dm_vp))")
    misfit_dvp = with_logger(ConsoleLogger(Logging.Error)) do
        swmisfit!(wavesim, matprop, shots;)
    end
    matprop.vp .-= eps_vp .* dm_vp

    matprop.rho .+= eps_rho .* dm_rho
    println("Computing ψ(ρ + ϵΔρ) where ϵ = $(eps_rho), norm(Δρ) = $(norm(dm_rho))")
    misfit_drho = with_logger(ConsoleLogger(Logging.Error)) do
        swmisfit!(wavesim, matprop, shots;)
    end
    matprop.rho .-= eps_rho .* dm_rho

    lhs_vp = dot(vec(grad["vp"]), vec(dm_vp))
    rhs_vp = (misfit_dvp - misfit) / eps_vp
    println("⟨∂ψ/∂vp, Δvp⟩ = $lhs_vp")
    println("(ψ(vp + ϵΔvp⟩) - ψ(vp)) / ϵ = $rhs_vp")
    println("Absolute error: $(lhs_vp - rhs_vp)")
    println("Relative error = $(abs(lhs_vp - rhs_vp) / abs(lhs_vp))")

    lhs_rho = dot(vec(grad["rho"]), vec(dm_rho))
    rhs_rho = (misfit_drho - misfit) / eps_rho
    println("⟨∂ψ/∂ρ, Δρ⟩ = $lhs_rho")
    println("(ψ(ρ + ϵΔρ⟩) - ψ(ρ)) / ϵ = $rhs_rho")
    println("Absolute error: $(lhs_rho - rhs_rho)")
    println("Relative error = $(abs(lhs_rho - rhs_rho) / abs(lhs_rho))")

    return grad, fdgrad_vp, fdgrad_rho
end

function check_elastic_iso(params, matprop, shots; computefd=false)
    # Build wavesim
    wavesim = build_wavesim(params, matprop; gradient=true, smooth_radius=0)
    # Compute gradients
    println("Computing adjoint ∂ψ/∂m and ψ(m)")
    grad, misfit = swgradient!(wavesim, matprop, shots; compute_misfit=true)

    eps_λ = minimum(matprop.λ) / 1e8
    eps_μ = minimum(matprop.μ) / 1e8
    eps_ρ = minimum(matprop.ρ) / 1e8
    # Compute FD gradient
    if computefd
        println("Computing FD ∂ψ/∂m with respect to λ")
        fdgrad = zeros(size(matprop.λ)...)
        dm = eps_λ
        for i in eachindex(matprop.λ)
            matprop.λ[i] += dm
            println("Computing FD gradient for index $i / $(length(matprop.vp))")
            with_logger(ConsoleLogger(Logging.Error)) do
                fdgrad[i] = (swmisfit!(wavesim, matprop, shots;) - misfit) / dm
            end
            matprop.λ[i] -= dm
        end
        serialize("fdgrad_lambda.jls", fdgrad)

        println("Computing FD ∂ψ/∂m with respect to μ")
        fdgrad = zeros(size(matprop.μ)...)
        dm = eps_μ
        for i in eachindex(matprop.μ)
            matprop.μ[i] += dm
            println("Computing FD gradient for index $i / $(length(matprop.vp))")
            with_logger(ConsoleLogger(Logging.Error)) do
                fdgrad[i] = (swmisfit!(wavesim, matprop, shots;) - misfit) / dm
            end
            matprop.μ[i] -= dm
        end
        serialize("fdgrad_mu.jls", fdgrad)

        println("Computing FD ∂ψ/∂m with respect to ρ")
        fdgrad = zeros(size(matprop.ρ)...)
        dm = eps_ρ
        for i in eachindex(matprop.ρ)
            matprop.ρ[i] += dm
            println("Computing FD gradient for index $i / $(length(matprop.vp))")
            with_logger(ConsoleLogger(Logging.Error)) do
                fdgrad[i] = (swmisfit!(wavesim, matprop, shots;) - misfit) / dm
            end
            matprop.ρ[i] -= dm
        end
        serialize("fdgrad_rho.jls", fdgrad)
    else
        if isfile("fdgrad_lambda.jls") && isfile("fdgrad_mu.jls") && isfile("fdgrad_rho.jls")
            println("Loading FD gradient from file")
            fdgrad_λ = deserialize("fdgrad_lambda.jls")
            fdgrad_μ = deserialize("fdgrad_mu.jls")
            fdgrad_ρ = deserialize("fdgrad_rho.jls")
        else
            println("FD gradient file not found. Set computefd=true to compute it.")
            fdgrad_λ = zeros(size(matprop.λ)...)
            fdgrad_μ = zeros(size(matprop.μ)...)
            fdgrad_ρ = zeros(size(matprop.ρ)...)
        end
    end
    # Compute dot product test
    halo = params.boundcond.halo
    dm_λ = rand(size(matprop.λ)...)
    dm_λ[1:halo, :] .= 0.0
    dm_λ[end-halo:end, :] .= 0.0
    # dm_λ[:, 1:halo] .= 0.0
    dm_λ[:, end-halo:end] .= 0.0
    dm_λ = dm_λ ./ norm(dm_λ)
    dm_μ = rand(size(matprop.μ)...)
    dm_μ[1:halo, :] .= 0.0
    dm_μ[end-halo:end, :] .= 0.0
    # dm_μ[:, 1:halo] .= 0.0
    dm_μ[:, end-halo:end] .= 0.0
    dm_μ = dm_μ ./ norm(dm_μ)
    dm_ρ = rand(size(matprop.ρ)...)
    dm_ρ[1:halo, :] .= 0.0
    dm_ρ[end-halo:end, :] .= 0.0
    # dm_ρ[:, 1:halo] .= 0.0
    dm_ρ[:, end-halo:end] .= 0.0
    dm_ρ = dm_ρ ./ norm(dm_ρ)

    matprop.λ .+= eps_λ .* dm_λ
    println("Computing ψ(λ + ϵΔλ) where ϵ = $eps_λ, norm(Δλ) = $(norm(dm_λ))")
    misfit_dλ = with_logger(ConsoleLogger(Logging.Error)) do
        swmisfit!(wavesim, matprop, shots;)
    end
    matprop.λ .-= eps_λ .* dm_λ

    matprop.μ .+= eps_μ .* dm_μ
    println("Computing ψ(μ + ϵΔμ) where ϵ = $eps_μ, norm(Δμ) = $(norm(dm_μ))")
    misfit_dμ = with_logger(ConsoleLogger(Logging.Error)) do
        swmisfit!(wavesim, matprop, shots;)
    end
    matprop.μ .-= eps_μ .* dm_μ

    matprop.ρ .+= eps_ρ .* dm_ρ
    println("Computing ψ(ρ + ϵΔρ) where ϵ = $eps_ρ, norm(Δρ) = $(norm(dm_ρ))")
    misfit_dρ = with_logger(ConsoleLogger(Logging.Error)) do
        swmisfit!(wavesim, matprop, shots;)
    end
    matprop.ρ .-= eps_ρ .* dm_ρ

    lhs_λ = dot(vec(grad["lambda"]), vec(dm_λ))
    rhs_λ = (misfit_dλ - misfit) / eps_λ
    println("⟨∂ψ/∂λ, Δλ⟩ = $lhs_λ")
    println("(ψ(λ + ϵΔλ⟩) - ψ(λ)) / ϵ = $rhs_λ")
    println("Absolute error: $(lhs_λ - rhs_λ)")
    println("Relative error = $(abs(lhs_λ - rhs_λ) / abs(lhs_λ))")

    lhs_μ = dot(vec(grad["mu"]), vec(dm_μ))
    rhs_μ = (misfit_dμ - misfit) / eps_μ
    println("⟨∂ψ/∂μ, Δμ⟩ = $lhs_μ")
    println("(ψ(μ + ϵΔμ⟩) - ψ(μ)) / ϵ = $rhs_μ")
    println("Absolute error: $(lhs_μ - rhs_μ)")
    println("Relative error = $(abs(lhs_μ - rhs_μ) / abs(lhs_μ))")

    lhs_ρ = dot(vec(grad["rho"]), vec(dm_ρ))
    rhs_ρ = (misfit_dρ - misfit) / eps_ρ
    println("⟨∂ψ/∂ρ, Δρ⟩ = $lhs_ρ")
    println("(ψ(ρ + ϵΔρ⟩) - ψ(ρ)) / ϵ = $rhs_ρ")
    println("Absolute error: $(lhs_ρ - rhs_ρ)")
    println("Relative error = $(abs(lhs_ρ - rhs_ρ) / abs(lhs_ρ))")

    return grad, fdgrad_λ, fdgrad_μ, fdgrad_ρ
end


# Check 1D CD
begin
    params, matprop, times, gridx, shots, srcstf = setup1D_CD()
    grad, fdgrad = check_acoustic_CD(params, matprop, shots)
    begin
        fig = Figure(size=(800, 1000))
        ax = Axis(fig[1, 1], title="Source time function", xlabel="Time (s)", ylabel="Amplitude")
        ax2 = Axis(fig[2, 1], title="Receiver seismogram", xlabel="Time (s)", ylabel="Pressure [Pa]")
        ax3 = Axis(fig[3, 1], title="Velocity model", xlabel="Distance (m)", ylabel="Velocity [m/s]")
        ax4 = Axis(fig[4, 1], title="Gradients", xlabel="Distance (m)", ylabel="∂ψ/∂m")
        # ax5 = Axis(fig[5, 1], title="Adjoint vs. FD gradient difference", xlabel="Distance (m)", ylabel="Absolute error")
        lines!(ax, times, srcstf[:, 1])
        for i in 1:size(shots[1].recs.seismograms, 2)
            lines!(ax2, times, shots[1].recs.seismograms[:, i]; label="Receiver $(i)")
        end
        axislegend(ax2)
        lines!(ax3, gridx, matprop.vp, label="Velocity model")
        lines!(ax4, gridx, grad["vp"], label="Adj gradient", color=:black)
        # lines!(ax4, gridx, fdgrad, label="FD gradient", color=:red, linestyle=:dash)
        scatter!(ax4, shots[1].srcs.positions[:, 1], zeros(1); marker=:star5, color=:red, markersize=20, label="Sources")
        scatter!(ax4, shots[1].recs.positions[:, 1], zeros(1); marker=:dtriangle, color=:black, markersize=20, label="Receivers")
        axislegend(ax4)
        # lines!(ax5, gridx, abs.(grad["vp"] .- fdgrad), label="Adjoint vs. FD gradient absolute error", color=:black)
        fig
    end
end

# Check 2D CD
begin
    params, matprop, times, gridx, gridy, shots, srcstf = setup2D_CD()
    grad, fdgrad = check_acoustic_CD(params, matprop, shots)
    begin
        fig = Figure(size=(800, 1000))
        ax = Axis(fig[1, 1:2], title="Source time function", xlabel="Time (s)", ylabel="Amplitude")
        ax2 = Axis(fig[2, 1:2], title="Receiver seismogram", xlabel="Time (s)", ylabel="Pressure [Pa]")
        ax3 = Axis(fig[3, 1], title="Velocity model", xlabel="Distance (m)", ylabel="Depth (m)", aspect=DataAspect())
        ax4 = Axis(fig[3, 2], title="Gradients", xlabel="Distance (m)", ylabel="Depth (m)", aspect=DataAspect())
        # ax5 = Axis(fig[5, 1], title="Adjoint vs. FD gradient difference", xlabel="Distance (m)", ylabel="Absolute error")
        lines!(ax, times, srcstf[:, 1])
        for i in 1:size(shots[1].recs.seismograms, 2)
            lines!(ax2, times, shots[1].recs.seismograms[:, i]; label="Receiver $(i)")
        end
        axislegend(ax2)
        hm = heatmap!(ax3, gridx, gridy, matprop.vp, colormap=Reverse(:batlowK))
        Colorbar(fig[4, 1], hm, label="Velocity [m/s]", vertical = false, flipaxis = false)
        ax3.yreversed = true
        clim = maximum(abs.(grad["vp"])) / 5
        hm = heatmap!(ax4, gridx, gridy, grad["vp"], colormap=:vik, colorrange=(-clim, clim), highclip = ColorSchemes.vik[end], lowclip = ColorSchemes.vik[1])
        Colorbar(fig[4, 2], hm, label="∂ψ/∂m", vertical = false, flipaxis = false)
        ax4.yreversed = true
        scatter!(ax3, shots[1].srcs.positions; marker=:star5, color=:red, markersize=20, label="Sources")
        scatter!(ax3, shots[1].recs.positions; marker=:dtriangle, color=:black, markersize=20, label="Receivers")
        scatter!(ax4, shots[1].srcs.positions; marker=:star5, color=:red, markersize=20, label="Sources")
        scatter!(ax4, shots[1].recs.positions; marker=:dtriangle, color=:black, markersize=20, label="Receivers")
        axislegend(ax4)
        fig
    end
end

# Check 2D VD
begin
    params, matprop, times, gridx, gridy, shots, srcstf = setup2D_VD()
    grad, fdgrad_vp, fdgrad_rho = check_acoustic_VD(params, matprop, shots)
    begin
        fig = Figure(size=(800, 1200))
        ax = Axis(fig[1, 1:2], title="Source time function", xlabel="Time (s)", ylabel="Amplitude")
        ax2 = Axis(fig[2, 1:2], title="Receiver seismogram", xlabel="Time (s)", ylabel="Pressure [Pa]")
        ax3 = Axis(fig[3, 1], title="Velocity model", xlabel="Distance (m)", ylabel="Depth (m)", aspect=DataAspect())
        ax4 = Axis(fig[3, 2], title="Density model", xlabel="Distance (m)", ylabel="Depth (m)", aspect=DataAspect())
        ax5 = Axis(fig[5, 1], title="Velocity gradient", xlabel="Distance (m)", ylabel="Depth (m)", aspect=DataAspect())
        ax6 = Axis(fig[5, 2], title="Density gradient", xlabel="Distance (m)", ylabel="Depth (m)", aspect=DataAspect())
        lines!(ax, times, srcstf[:, 1])
        for i in 1:size(shots[1].recs.seismograms, 2)
            lines!(ax2, times, shots[1].recs.seismograms[:, i]; label="Receiver $(i)")
        end
        axislegend(ax2)
        hm = heatmap!(ax3, gridx, gridy, matprop.vp, colormap=Reverse(:batlowK))
        Colorbar(fig[4, 1], hm, label="Velocity [m/s]", vertical = false, flipaxis = false)
        ax3.yreversed = true
        hm = heatmap!(ax4, gridx, gridy, matprop.rho, colormap=Reverse(:batlowK))
        Colorbar(fig[4, 2], hm, label="Density [kg/m³]", vertical = false, flipaxis = false)
        ax4.yreversed = true
        clim = maximum(abs.(grad["vp"])) / 5
        hm = heatmap!(ax5, gridx, gridy, grad["vp"], colormap=:vik, colorrange=(-clim, clim), highclip = ColorSchemes.vik[end], lowclip = ColorSchemes.vik[1])
        Colorbar(fig[6, 1], hm, label="∂ψ/∂vₚ", vertical = false, flipaxis = false)
        ax5.yreversed = true
        clim = maximum(abs.(grad["rho"])) / 5
        hm = heatmap!(ax6, gridx, gridy, grad["rho"], colormap=:vik, colorrange=(-clim, clim), highclip = ColorSchemes.vik[end], lowclip = ColorSchemes.vik[1])
        Colorbar(fig[6, 2], hm, label="∂ψ/∂ρ", vertical = false, flipaxis = false)
        ax6.yreversed = true
        scatter!(ax3, shots[1].srcs.positions; marker=:star5, color=:red, markersize=20, label="Sources")
        scatter!(ax3, shots[1].recs.positions; marker=:dtriangle, color=:black, markersize=20, label="Receivers")
        scatter!(ax4, shots[1].srcs.positions; marker=:star5, color=:red, markersize=20, label="Sources")
        scatter!(ax4, shots[1].recs.positions; marker=:dtriangle, color=:black, markersize=20, label="Receivers")
        scatter!(ax5, shots[1].srcs.positions; marker=:star5, color=:red, markersize=20, label="Sources")
        scatter!(ax5, shots[1].recs.positions; marker=:dtriangle, color=:black, markersize=20, label="Receivers")
        scatter!(ax6, shots[1].srcs.positions; marker=:star5, color=:red, markersize=20, label="Sources")
        scatter!(ax6, shots[1].recs.positions; marker=:dtriangle, color=:black, markersize=20, label="Receivers")
        axislegend(ax6)
        fig
    end
end

# Check 2D isoelastic
params, matprop, times, gridx, gridy, shots, srcstf, momten = setup2D_isoela()
grad, fdgrad_λ, fdgrad_μ, fdgrad_ρ = check_elastic_iso(params, matprop, shots)
begin
    fig = Figure(size=(1000, 1200))
    ax = Axis(fig[1, 1:3], title="Source time function", xlabel="Time (s)", ylabel="Amplitude")
    ax2 = Axis(fig[2, 1:3], title="Receiver seismogram", xlabel="Time (s)", ylabel="Velocity [m/s]")
    ax3 = Axis(fig[3, 1], title="λ model", xlabel="Distance (m)", ylabel="Depth (m)", aspect=DataAspect())
    ax4 = Axis(fig[3, 2], title="μ model", xlabel="Distance (m)", ylabel="Depth (m)", aspect=DataAspect())
    ax5 = Axis(fig[3, 3], title="ρ model", xlabel="Distance (m)", ylabel="Depth (m)", aspect=DataAspect())
    ax6 = Axis(fig[5, 1], title="λ gradient", xlabel="Distance (m)", ylabel="Depth (m)", aspect=DataAspect())
    ax7 = Axis(fig[5, 2], title="μ gradient", xlabel="Distance (m)", ylabel="Depth (m)", aspect=DataAspect())
    ax8 = Axis(fig[5, 3], title="ρ gradient", xlabel="Distance (m)", ylabel="Depth (m)", aspect=DataAspect())
    lines!(ax, times, srcstf[:, 1])
    for i in 1:size(shots[1].recs.seismograms, 3)
        lines!(ax2, times, shots[1].recs.seismograms[:, 1, i]; label="Receiver vx $(i)")
        lines!(ax2, times, shots[1].recs.seismograms[:, 2, i]; label="Receiver vz $(i)")
    end
    axislegend(ax2)

    hm = heatmap!(ax3, gridx, gridy, matprop.λ, colormap=Reverse(:batlowK))
    Colorbar(fig[4, 1], hm, label="λ [Pa]", vertical = false, flipaxis = false)
    ax3.yreversed = true

    hm = heatmap!(ax4, gridx, gridy, matprop.μ, colormap=Reverse(:batlowK))
    Colorbar(fig[4, 2], hm, label="μ [Pa]", vertical = false, flipaxis = false)
    ax4.yreversed = true

    hm = heatmap!(ax5, gridx, gridy, matprop.ρ, colormap=Reverse(:batlowK))
    Colorbar(fig[4, 3], hm, label="ρ [kg/m³]", vertical = false, flipaxis = false)
    ax5.yreversed = true

    clim = maximum(abs.(grad["lambda"])) / 1e2
    hm = heatmap!(ax6, gridx, gridy, grad["lambda"], colormap=:vik, colorrange=(-clim, clim), highclip = ColorSchemes.vik[end], lowclip = ColorSchemes.vik[1])
    Colorbar(fig[6, 1], hm, label="∂ψ/∂λ", vertical = false, flipaxis = false)
    ax6.yreversed = true

    clim = maximum(abs.(grad["mu"])) / 1e2
    hm = heatmap!(ax7, gridx, gridy, grad["mu"], colormap=:vik, colorrange=(-clim, clim), highclip = ColorSchemes.vik[end], lowclip = ColorSchemes.vik[1])
    Colorbar(fig[6, 2], hm, label="∂ψ/∂μ", vertical = false, flipaxis = false)
    ax7.yreversed = true

    clim = maximum(abs.(grad["rho"])) / 5
    hm = heatmap!(ax8, gridx, gridy, grad["rho"], colormap=:vik, colorrange=(-clim, clim), highclip = ColorSchemes.vik[end], lowclip = ColorSchemes.vik[1])
    Colorbar(fig[6, 3], hm, label="∂ψ/∂ρ", vertical = false, flipaxis = false)
    ax8.yreversed = true

    # hm = heatmap!(ax4, gridx, gridy, matprop.rho, colormap=Reverse(:batlowK))
    # Colorbar(fig[4, 2], hm, label="Density [kg/m³]", vertical = false, flipaxis = false)
    # ax4.yreversed = true
    # clim = maximum(abs.(grad["vp"])) / 5
    # hm = heatmap!(ax5, gridx, gridy, grad["vp"], colormap=:vik, colorrange=(-clim, clim), highclip = ColorSchemes.vik[end], lowclip = ColorSchemes.vik[1])
    # Colorbar(fig[6, 1], hm, label="∂ψ/∂vₚ", vertical = false, flipaxis = false)
    # ax5.yreversed = true
    # clim = maximum(abs.(grad["rho"])) / 5
    # hm = heatmap!(ax6, gridx, gridy, grad["rho"], colormap=:vik, colorrange=(-clim, clim), highclip = ColorSchemes.vik[end], lowclip = ColorSchemes.vik[1])
    # Colorbar(fig[6, 2], hm, label="∂ψ/∂ρ", vertical = false, flipaxis = false)
    # ax6.yreversed = true
    # scatter!(ax3, shots[1].srcs.positions; marker=:star5, color=:red, markersize=20, label="Sources")
    # scatter!(ax3, shots[1].recs.positions; marker=:dtriangle, color=:black, markersize=20, label="Receivers")
    # scatter!(ax4, shots[1].srcs.positions; marker=:star5, color=:red, markersize=20, label="Sources")
    # scatter!(ax4, shots[1].recs.positions; marker=:dtriangle, color=:black, markersize=20, label="Receivers")
    # scatter!(ax5, shots[1].srcs.positions; marker=:star5, color=:red, markersize=20, label="Sources")
    # scatter!(ax5, shots[1].recs.positions; marker=:dtriangle, color=:black, markersize=20, label="Receivers")
    # scatter!(ax6, shots[1].srcs.positions; marker=:star5, color=:red, markersize=20, label="Sources")
    # scatter!(ax6, shots[1].recs.positions; marker=:dtriangle, color=:black, markersize=20, label="Receivers")
    # axislegend(ax6)
    fig
end