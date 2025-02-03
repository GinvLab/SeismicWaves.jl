using Revise
using SeismicWaves
using LinearAlgebra
using GLMakie
using Logging
using Serialization
using Printf

function gaussian_vel_2D(nx, ny, c0, c0max, r, origin=[(nx + 1) / 2, (ny + 1) / 2])
    sigma = r / 3
    amp = c0max - c0
    f(x, y) = amp * exp(-(0.5 * ((x - origin[1])^2 + (y - origin[2])^2) / sigma^2))
    return c0 .+ [f(x, y) for x in 1:nx, y in 1:ny]
end

function gradient_vel_2D(nx, ny, c0, c0max)
    c = zeros(nx, ny)
    dc = (c0max - c0) / (ny-1)
    for j in 1:ny
        c[:, j] .= c0 .+ dc * (j-1)
    end
    return c
end

function create_constant_velocity_model(nx, nz, ρ0, vp0, vs0=vp0 / sqrt(3))
    vp = vp0 .* ones(nx, nz)
    vs = vs0 .* ones(nx, nz)
    ρ = ρ0 .* ones(nx, nz)
    μ = vs .^ 2 .* ρ 
    λ = (vp .^ 2 .* ρ) .- (2 .* μ)
    matprop = ElasticIsoMaterialProperties(; λ=λ, μ=μ, ρ=ρ)
    return matprop
end

function create_gaussian_velocity_model(nx, nz, ρ0, vp0, vs0=vp0 / sqrt(3))
    vp = gaussian_vel_2D(nx, nz, vp0, vp0 * 1.2, nz/5)
    vs = vs0 .* ones(nx, nz)
    ρ = ρ0 .* ones(nx, nz)
    μ = vs .^ 2 .* ρ 
    λ = (vp .^ 2 .* ρ) .- (2 .* μ)
    matprop = ElasticIsoMaterialProperties(; λ=λ, μ=μ, ρ=ρ)
    return matprop
end

# Function to define shots
function define_shots(lx, lz, nx, nz, nt, dt, dh)
    nsrc = 1
    possrcs = zeros(nsrc, 2)
    possrcs[1, :] .= [lx / 3, lz / 2]
    f0 = 2.0
    t0 = 1.20 / f0
    srcstf = zeros(nt, nsrc)
    Mxx = fill(1e15, nsrc)
    Mzz = fill(1e15, nsrc)
    Mxz = fill(0.0, nsrc)
    srcstf[:, 1] .= rickerstf.(collect(Float64, range(0.0; step=dt, length=nt)), t0, f0)
    # srcstf = zeros(nt, 2, nsrc)
    # for s in 1:nsrc
    #     srcstf[:, 1, s] .= 1e10 .* rickerstf.(collect(Float64, range(0.0; step=dt, length=nt)), t0, f0)
    #     srcstf[:, 2, s] .= 1e10 .* rickerstf.(collect(Float64, range(0.0; step=dt, length=nt)), t0, f0)
    # end
    # srcs = ExternalForceSources(possrcs, srcstf, f0)
    srcs = MomentTensorSources(possrcs, srcstf, [MomentTensor2D(; Mxx=Mxx[s], Mzz=Mzz[s], Mxz=Mxz[s]) for s in 1:nsrc], f0)
    nrecs = 1
    posrecs = zeros(nrecs, 2)
    posrecs[1, :] .= [2lx / 3, lz / 2]

    recs = VectorReceivers(posrecs, nt)
    # shots = [ExternalForceShot(; srcs=srcs, recs=recs)]
    shots = [MomentTensorShot(; srcs=srcs, recs=recs)]
    return shots
end

# Function to compute the forward simulation
function compute_forward_simulation(params, matprop, shots)
    model = build_wavesim(params, matprop; parall=:threads, gradient=true, check_freq=nothing, snapevery=20)
    snaps = swforward!(model, matprop, shots)
    return model, snaps
end

# Function to compute the gradient
function compute_gradient(model, matprop_grad, shots_grad)
    grad, misfit = swgradient!(model, matprop_grad, shots_grad; compute_misfit=true)
    return grad, misfit
end

# Function to compute the FD gradient
function compute_fd_gradient!(grad_FD, param_name, Δparam, model, matprop_grad, shots_grad, misfit, halo, nx, nz)
    for i in halo:nx-halo
        for j in halo:nz-halo
            new_param = copy(getfield(matprop_grad, param_name))
            new_param[i, j] += Δparam
            matpropnew = ElasticIsoMaterialProperties(; ρ=matprop_grad.ρ, μ=matprop_grad.μ, λ=matprop_grad.λ)
            setfield!(matpropnew, param_name, new_param)
            misfit_right = swmisfit!(model, matpropnew, shots_grad; logger=ConsoleLogger(stderr, Logging.Warn))
            grad_FD[i, j] = (misfit_right - misfit) / Δparam
            @printf("Parameter: %s, Indices: (%d, %d), Gradient: %.6e\n", param_name, i, j, grad_FD[i, j])
        end
    end
end

# Main function to orchestrate the workflow
function main()
    # Time and grid parameters
    halo = 20
    rcoef = 0.0001
    nx, nz = 11*7*2+1 + 2halo, 11*7*2+1 + 2halo
    vp0 = 3100.0
    vs0 = vp0 / sqrt(3)
    # vs0 = 0.0
    ρ0 = 2700.0
    lx = lz = 6000.0
    dx, dz = lx / (nx - 1), lz / (nz - 1)
    T = 5.0
    dt = dx / vp0 / sqrt(2) * 0.7
    nt = ceil(Int, T / dt)
    dh = dx = dz

    # Create velocity model
    matprop_const = create_constant_velocity_model(nx, nz, ρ0, vp0, vs0)
    # matprop_const.λ .= gradient_vel_2D(nx, nz, λ0, λ0 * 1.15)
    matprop_gauss = deepcopy(matprop_const)
    matprop_gauss.ρ .*= 0.9
    # matprop_gauss.ρ .= gradient_vel_2D(nx, nz, ρ0, ρ0 * 0.75)
    # matprop_gauss = create_constant_velocity_model(nx, nz, ρ0 * 0.75, vp0, vs0)


    # Define shots
    shots = define_shots(lx, lz, nx, nz, nt, dt, dh)

    # Input parameters for elastic simulation
    boundcond = CPMLBoundaryConditionParameters(; halo=halo, rcoef=rcoef, freeboundtop=false)
    params = InputParametersElastic(nt, dt, (nx, nz), (dh, dh), boundcond)

    # Compute forward simulation
    println("Computing forward simulation to get observed data")
    model, snapshots = compute_forward_simulation(params, matprop_gauss, shots)

    # Compute the gradient
    # shots_grad = Vector{ExternalForceShot{Float64, 2}}()
    shots_grad = Vector{MomentTensorShot{Float64, 2, MomentTensor2D{Float64}}}()
    for i in eachindex(shots)
        recs_grad = VectorReceivers(shots[i].recs.positions, nt; observed=copy(shots[i].recs.seismograms), invcov=1.0 * I(nt))
        # push!(shots_grad, ExternalForceShot(; srcs=shots[i].srcs, recs=recs_grad))
        push!(shots_grad, MomentTensorShot(; srcs=shots[i].srcs, recs=recs_grad))
    end

    println("Compute gradient with adjoint method")
    grad, misfit = compute_gradient(model, matprop_const, shots_grad)
    serialize("grads_adj.jls", grad)

    # # Check gradient with respect to density
    # println("Check gradient with respect to ρ")
    # @show minimum(matprop_const.ρ)
    # Δρ = rand(nx, nz) .* minimum(matprop_const.ρ) .* 1e-6
    # FD_matprop_right = ElasticIsoMaterialProperties(; ρ=matprop_const.ρ .+ Δρ, μ=matprop_const.μ, λ=matprop_const.λ)
    # FD_matprop_left = ElasticIsoMaterialProperties(; ρ=matprop_const.ρ .- Δρ, μ=matprop_const.μ, λ=matprop_const.λ)
    # FD_misfit_right = swmisfit!(params, FD_matprop_right, shots_grad)
    # FD_misfit_left = swmisfit!(params, FD_matprop_left, shots_grad)
    # right = FD_misfit_right - FD_misfit_left
    # left = dot(reduce(vcat, grad["rho"]), reduce(vcat, 2 .* Δρ))
    # println("χ(m + Δρ) - χ(m - Δρ) = ", right)
    # println("∂χ/∂ρ * 2Δρ = ", left)
    # println("Difference = ", right - left)
    # println("Ratio = ", left / right)

    # # Check gradient with respect to λ
    # println("Check gradient with respect to λ")
    # @show minimum(matprop_const.λ)
    # Δλ = rand(nx, nz) .* minimum(matprop_const.λ) .* 1e-6
    # FD_matprop_right = ElasticIsoMaterialProperties(; ρ=matprop_const.ρ, μ=matprop_const.μ, λ=matprop_const.λ .+ Δλ)
    # FD_matprop_left = ElasticIsoMaterialProperties(; ρ=matprop_const.ρ, μ=matprop_const.μ, λ=matprop_const.λ .- Δλ)
    # FD_misfit_right = swmisfit!(params, FD_matprop_right, shots_grad)
    # FD_misfit_left = swmisfit!(params, FD_matprop_left, shots_grad)
    # right = FD_misfit_right - FD_misfit_left
    # left = dot(reduce(vcat, grad["lambda"]), reduce(vcat, 2 .* Δλ))
    # println("χ(m + Δλ) - χ(m - Δλ) = ", right)
    # println("∂χ/∂λ * 2Δλ = ", left)
    # println("Difference = ", right - left)
    # println("Ratio = ", left / right)

    # println("Check gradient with respect to λ")
    # @show minimum(matprop_const.λ)
    # Δλ = rand(nx, nz) .* minimum(matprop_const.λ) .* 1e-6
    # FD_matprop_right = ElasticIsoMaterialProperties(; ρ=matprop_const.ρ, μ=matprop_const.μ, λ=matprop_const.λ .+ Δλ)
    # FD_matprop_left = ElasticIsoMaterialProperties(; ρ=matprop_const.ρ, μ=matprop_const.μ, λ=matprop_const.λ .- Δλ)
    # FD_misfit_right = swmisfit!(params, FD_matprop_right, shots_grad)
    # FD_misfit_left = swmisfit!(params, FD_matprop_left, shots_grad)
    # right = FD_misfit_right - FD_misfit_left
    # left = dot(reduce(vcat, grad["lambda"]), reduce(vcat, 2 .* Δλ))
    # println("χ(m + Δλ) - χ(m - Δλ) = ", right)
    # println("∂χ/∂λ * 2Δλ = ", left)
    # println("Difference = ", right - left)
    # println("Ratio = ", left / right)

    # # Check gradient with respect to μ
    # println("Check gradient with respect to μ")
    # @show minimum(matprop_const.μ)
    # Δμ = rand(nx, nz) .* minimum(matprop_const.μ) .* 1e-6
    # FD_matprop_right = ElasticIsoMaterialProperties(; ρ=matprop_const.ρ, μ=matprop_const.μ .+ Δμ, λ=matprop_const.λ)
    # FD_matprop_left = ElasticIsoMaterialProperties(; ρ=matprop_const.ρ, μ=matprop_const.μ .- Δμ, λ=matprop_const.λ)
    # FD_misfit_right = swmisfit!(params, FD_matprop_right, shots_grad)
    # FD_misfit_left = swmisfit!(params, FD_matprop_left, shots_grad)
    # right = FD_misfit_right - FD_misfit_left
    # left = dot(reduce(vcat, grad["mu"]), reduce(vcat, 2 .* Δμ))
    # println("χ(m + Δμ) - χ(m - Δμ) = ", right)
    # println("∂χ/∂μ * 2Δμ = ", left)
    # println("Difference = ", right - left)
    # println("Ratio = ", left / right)

    model = build_wavesim(params, matprop_const; parall=:threads)
    matprop_grad = deepcopy(matprop_const)
    # Compute FD gradients
    Δρ = minimum(matprop_const.ρ) * 1e-4
    grad_ρ_FD = zeros(nx, nz)
    compute_fd_gradient!(grad_ρ_FD, :ρ, Δρ, model, matprop_grad, shots_grad, misfit, halo, nx, nz)
    serialize("grad_ρ_FD.jls", grad_ρ_FD)

    Δλ = minimum(matprop_const.λ) * 1e-4
    grad_λ_FD = zeros(nx, nz)
    compute_fd_gradient!(grad_λ_FD, :λ, Δλ, model, matprop_grad, shots_grad, misfit, halo, nx, nz)
    serialize("grad_λ_FD.jls", grad_λ_FD)

    Δμ = minimum(matprop_const.μ) * 1e-4
    grad_μ_FD = zeros(nx, nz)
    compute_fd_gradient!(grad_μ_FD, :μ, Δμ, model, matprop_grad, shots_grad, misfit, halo, nx, nz)
    serialize("grad_μ_FD.jls", grad_μ_FD)

    return params, matprop_gauss, shots, snapshots
end

# Function to load and plot gradients
function load_and_plot_gradients(params, shots)
    grads = deserialize("grads_adj.jls")
    # grad_ρ_FD = deserialize("grad_ρ_FD.jls")
    # grad_λ_FD = deserialize("grad_λ_FD.jls")
    # grad_μ_FD = deserialize("grad_μ_FD.jls")
    grad_ρ_FD = grad_λ_FD = grad_μ_FD = zeros(params.gridsize...)
    # Plot gradients
    plotgrad(params, grads, shots, grad_ρ_FD, grad_λ_FD, grad_μ_FD)
end

function plotgrad(par, grad, shots, grad_ρ_FD, grad_λ_FD, grad_μ_FD)
    xgrd = [par.gridspacing[1] * (i - 1) for i in 1:par.gridsize[1]]
    ygrd = [par.gridspacing[2] * (i - 1) for i in 1:par.gridsize[2]]

    fig = Figure(; size=(500, 1000))

    axes = [
        (Axis(fig[1, 1]; aspect=DataAspect(), title="grad ρ", xlabel="x [m]", ylabel="z [m]"), grad["rho"], "∂χ/∂ρ", [1,2]),
        # (Axis(fig[1, 3]; aspect=DataAspect(), title="grad ρ (FD)", xlabel="x [m]", ylabel="z [m]"), grad_ρ_FD, "∂χ/∂ρ", [1,4]),
        (Axis(fig[2, 1]; aspect=DataAspect(), title="grad λ", xlabel="x [m]", ylabel="z [m]"), grad["lambda"], "∂χ/∂λ", [2,2]),
        # (Axis(fig[2, 3]; aspect=DataAspect(), title="grad λ (FD)", xlabel="x [m]", ylabel="z [m]"), grad_λ_FD, "∂χ/∂λ", [2,4]),
        (Axis(fig[3, 1]; aspect=DataAspect(), title="grad μ", xlabel="x [m]", ylabel="z [m]"), grad["mu"], "∂χ/∂μ", [3,2]),
        # (Axis(fig[3, 3]; aspect=DataAspect(), title="grad μ (FD)", xlabel="x [m]", ylabel="z [m]"), grad_μ_FD, "∂χ/∂μ", [3,4])
    ]

    radius = 5

    for (ax, grad_data, label, cbarpos) in axes
        grad2 = copy(grad_data)
        for s in 1:size(shots[1].srcs.positions, 1)
            srcx, srcy = ceil.(Int, shots[1].srcs.positions[s, :] ./ par.gridspacing) .+ 1
            grad2[srcx-radius:srcx+radius, srcy-radius:srcy+radius] .= 0
        end
        for r in 1:size(shots[1].recs.positions, 1)
            recx, recy = ceil.(Int, shots[1].recs.positions[r, :] ./ par.gridspacing) .+ 1
            grad2[recx-radius:recx+radius, recy-radius:recy+radius] .= 0
        end
        maxabsgrad = maximum(abs.(grad2))
        hm = heatmap!(ax, xgrd, ygrd, grad_data; colormap=:vik, colorrange=(-maxabsgrad, maxabsgrad))
        Colorbar(fig[cbarpos[1], cbarpos[2]], hm; label=label)
        ax.yreversed = true
    end

    save("grads.png", fig)
    return fig
end

function snapanimate(par, matprop, shots, snapsh; scalamp=0.01, snapevery=20)
    xgrd = [par.gridspacing[1] * (i - 1) for i in 1:par.gridsize[1]]
    ygrd = [par.gridspacing[2] * (i - 1) for i in 1:par.gridsize[2]]
    xrec = shots[1].recs.positions[:, 1]
    yrec = shots[1].recs.positions[:, 2]
    xsrc = shots[1].srcs.positions[:, 1]
    ysrc = shots[1].srcs.positions[:, 2]

    vxsnap = [snapsh[1][kk]["v"].value[1] for kk in sort(keys(snapsh[1]))]
    vzsnap = [snapsh[1][kk]["v"].value[2] for kk in sort(keys(snapsh[1]))]

    curvx = Observable(vxsnap[1])
    curvz = Observable(vzsnap[1])

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
    fig = Figure(; size=(1500, 500))

    nframes = length(vxsnap)

    cmapwavefield = :vik #:cyclic_grey_15_85_c0_n256_s25 #:balance

    ax1 = Axis(fig[1, 1]; aspect=DataAspect(),
        xlabel="x [m]", ylabel="z [m]", title="Vx, clip at $scalamp of max amplitude, iteration 0 of $(snapevery*nframes)")
    #poly!(ax4,Rect(rect...),color=:green,alpha=0.3)
    extx = extrema.([vxsnap[i] for i in eachindex(vxsnap)])
    extx = map(p -> max(abs(p[1]), abs(p[2])), extx)
    vmax = max(extx...)
    vminmax = scalamp .* (-vmax, vmax)
    hm = heatmap!(ax1, xgrd, ygrd, curvx; colormap=cmapwavefield,
        colorrange=vminmax) #,alpha=0.7)
    Colorbar(fig[1, 2], hm; label="x partic. vel.")

    lines!(ax1, Rect(rectpml...); color=:green)
    scatter!(ax1, xrec, yrec; marker=:dtriangle, label="Receivers", markersize=15)
    scatter!(ax1, xsrc, ysrc; label="Sources", markersize=15)
    # axislegend(ax1)
    ax1.yreversed = true

    ax2 = Axis(fig[1, 3]; aspect=DataAspect(),
        xlabel="x [m]", ylabel="z [m]", title="Vz, clip at $scalamp of max amplitude, iteration 0 of $(snapevery*nframes)")
    #poly!(ax4,Rect(rect...),color=:green,alpha=0.3)
    extx = extrema.([vzsnap[i] for i in eachindex(vzsnap)])
    extx = map(p -> max(abs(p[1]), abs(p[2])), extx)
    vmax = max(extx...)
    vminmax = scalamp .* (-vmax, vmax)
    hm = heatmap!(ax2, xgrd, ygrd, curvz; colormap=cmapwavefield,
        colorrange=vminmax) #,alpha=0.7)
    Colorbar(fig[1, 4], hm; label="z partic. vel.")

    lines!(ax2, Rect(rectpml...); color=:green)
    scatter!(ax2, xrec, yrec; marker=:dtriangle, label="Receivers", markersize=15)
    scatter!(ax2, xsrc, ysrc; label="Sources", markersize=15)
    # axislegend(ax2)
    ax2.yreversed = true

    ax3 = Axis(fig[1, 5]; aspect=DataAspect(),
        xlabel="x [m]", ylabel="z [m]")
    hm = heatmap!(ax3, xgrd, ygrd, vp; colormap=:Reds) #,alpha=0.7)
    Colorbar(fig[1, 6], hm; label="Vp")

    scatter!(ax3, xrec, yrec; marker=:dtriangle, label="Receivers", markersize=15)
    scatter!(ax3, xsrc, ysrc; label="Sources", markersize=15)
    axislegend(ax3)
    ax3.yreversed = true

    ##
    save("first_frame.png", fig)
    ##=====================================

    function updatefunction(curax1, curax2, vxsnap, vzsnap, it)
        cvx = vxsnap[it]
        cvz = vzsnap[it]
        curax1.title = "Vx, clip at $scalamp of max amplitude, iteration $(snapevery*it) of $(snapevery*nframes)"
        curax2.title = "Vz, clip at $scalamp of max amplitude, iteration $(snapevery*it) of $(snapevery*nframes)"
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

# Run the main function
params, matprop_gauss, shots, snapshots = main()
# Plot animation
snapanimate(params, matprop_gauss, shots, snapshots; scalamp=0.005)
# Load and plot gradients
load_and_plot_gradients(params, shots)
