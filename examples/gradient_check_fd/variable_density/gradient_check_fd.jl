
using SeismicWaves
using LinearAlgebra

###################################################################
using Logging
using Plots

using CUDA
using Serialization

info_logger = ConsoleLogger(stderr, Logging.Info)
warn_logger = ConsoleLogger(stderr, Logging.Warn)
error_logger = ConsoleLogger(stderr, Logging.Error)
debug_logger = ConsoleLogger(stderr, Logging.Debug)

include("../../models.jl")
include("../../geometries.jl")
include("../../plotting_utils.jl")

function setup(nt, c0, c0max, rho0, rho0max, r, dx, dy, dt, halo, rcoef, nx, ny, parall)
    ##========================================
    # time stuff
    t = collect(Float64, range(0.0; step=dt, length=nt)) # seconds

    ##========================================
    # create constant and gaussian velocity model
    lx = (nx - 1) * dx
    ly = (ny - 1) * dy
    matprop_const = VpRhoAcousticVDMaterialProperties(c0 .* ones(nx, ny), rho0 .* ones(nx, ny))
    # gaussian perturbed model
    matprop_gauss = VpRhoAcousticVDMaterialProperties(gaussian_vel_2D(nx, ny, c0, c0max, r), gaussian_vel_2D(nx, ny, rho0, rho0max, r))

    ##========================================
    # shots definition
    nshots = 10
    f0 = 10.0
    t0 = 4 / f0
    srctf = gaussdersource1D.(t, t0, f0)
    dd = 60
    linear_2D_geometry(
        nshots,
        matprop_gauss.vp,
        f0,
        nt,
        srctf,
        dd,
        lx,
        ly,
        dx,
        dy,
        halo;
        plot_geometry=true,
        save_file="setup_vp.png"
    )
    shots = linear_2D_geometry(
        nshots,
        matprop_gauss.rho,
        f0,
        nt,
        srctf,
        dd,
        lx,
        ly,
        dx,
        dy,
        halo;
        plot_geometry=true,
        save_file="setup_rho.png"
    )

    ##============================================
    ## Input parameters for acoustic simulation
    boundcond = CPMLBoundaryConditionParameters(; halo=halo, rcoef=rcoef, freeboundtop=false)
    params = InputParametersAcoustic(nt, dt, (nx, ny), (dx, dy), boundcond)

    # Wave simulation builder
    wavesim = build_wavesim(params, matprop_const; gradient=true, parall=parall, check_freq=ceil(Int, sqrt(nt)), smooth_radius=0)

    return wavesim, shots, matprop_const, matprop_gauss
end

function gradient_fd_check(wavesim, shots, matprop_const, matprop_gauss)
    println("Computing forward solver")

    # compute forward gaussian
    with_logger(info_logger) do
        swforward!(wavesim, matprop_gauss, shots)
    end

    # new receivers with observed seismograms
    shots_obs = Vector{ScalarShot{Float64}}()  #Pair{ScalarSources, ScalarReceivers}}()
    for i in eachindex(shots)
        # receivers definition
        recs = ScalarReceivers(
            copy(shots[i].recs.positions),
            nt;
            observed=copy(shots[i].recs.seismograms),
            invcov=Diagonal(ones(nt))
        )
        # add pair as shot
        push!(shots_obs, ScalarShot(; srcs=shots[i].srcs, recs=recs)) # srcs => recs)
    end

    println("Computing gradients")

    # compute gradients and misfit
    gradient, misfit = with_logger(info_logger) do
        swgradient!(wavesim, matprop_const, shots_obs; compute_misfit=true)
    end
    # save gradient
    serialize("grad.dat", reshape(hcat(gradient["vp"], gradient["rho"]), nx, ny, 2))

    println("Initial misfit: $misfit")

    # Compute gradients with finite differences
    fd_gradient_vp = zeros(nx, ny)
    dm_vp = -1e-3
    for i in 1:nx
        for j in 1:ny
            println("Computing ($i, $j) gradient wrt vp with FD")
            vp_perturbed = copy(matprop_const.vp)
            vp_perturbed[i, j] += dm_vp
            matprop_perturbed = VpRhoAcousticVDMaterialProperties(vp_perturbed, matprop_const.rho)
            @time new_misfit = with_logger(error_logger) do
                swmisfit!(wavesim, matprop_perturbed, shots_obs)
            end
            println("New misfit: $new_misfit")
            fd_gradient_vp[i, j] = (new_misfit - misfit) / dm_vp
        end
    end
    fd_gradient_rho = zeros(nx, ny)
    dm_rho = -1e-3
    for i in 1:nx
        for j in 1:ny
            println("Computing ($i, $j) gradient wrt rho with FD")
            rho_perturbed = copy(matprop_const.rho)
            rho_perturbed[i, j] += dm_rho
            matprop_perturbed = VpRhoAcousticVDMaterialProperties(matprop_const.vp, rho_perturbed)
            @time new_misfit = with_logger(error_logger) do
                swmisfit!(wavesim, matprop_perturbed, shots_obs)
            end
            println("New misfit: $new_misfit")
            fd_gradient_rho[i, j] = (new_misfit - misfit) / dm_rho
        end
    end

    fd_gradient = reshape(hcat(fd_gradient_vp, fd_gradient_rho), nx, ny, 2)

    # save FD gradient into file
    serialize("fdgrad.dat", fd_gradient)
    serialize("fdgrad_diff.dat", fd_gradient .- gradient)
end

function load_gradients()
    # deserialize saved gradients
    fdgrad = deserialize("fdgrad.dat")
    fdgrad_diff = deserialize("fdgrad_diff.dat")
    grad = fdgrad .- fdgrad_diff
    return grad, fdgrad, -fdgrad_diff
end

########################################################################

# Backend selection
parall = :GPU
device!(4)
run = true

# Numerical parameters
nt = 1000
c0 = 1000
c0max = 1300
rho0 = 1500
rho0max = 3000
r = 35
dh = dx = dy = 5.0
dt = dh / sqrt(2) / c0max * 6 / 7
halo = 20
rcoef = 0.0001
nx = 201
ny = 201
lx = (nx - 1) * dx
ly = (ny - 1) * dy

# setup
wavesim, shots, matprop_const, matprop_gauss = setup(nt, c0, c0max, rho0, rho0max, r, dx, dy, dt, halo, rcoef, nx, ny, parall)
if run
    gradient_fd_check(wavesim, shots, matprop_const, matprop_gauss)
end
# load saved results
adjgrad, fdgrad, grad_diff = load_gradients()

##################################################################

# corner1 = 21
# corner2 = round(Int, 300 ÷ dx)
# l = @layout([A B C])

# # Plot adjoint gradient and zoom in
# p_grad = plot_zoom(adjgrad, corner1, plot_nice_heatmap_grad; lx=lx, ly=ly, dx=dx, dy=dy)
# p_grad_zoom = plot_zoom(
#     adjgrad[corner1:end-corner1, corner1:end-corner1],
#     corner2,
#     plot_nice_heatmap_grad;
#     lx=(nx - corner1 * 2) * dx,
#     ly=(ny - corner1 * 2) * dy,
#     dx=dx,
#     dy=dy,
#     shift=-dx * corner1
# )
# p_grad_zoom2 = plot_nice_heatmap_grad(
#     adjgrad[corner2:end-corner2, corner2:end-corner2];
#     lx=(nx - corner2 * 2) * dx,
#     ly=(ny - corner2 * 2) * dy,
#     dx=dx,
#     dy=dy,
#     shift=-dx * corner2
# )
# p = plot(
#     p_grad,
#     p_grad_zoom,
#     p_grad_zoom2;
#     layout=l,
#     legend=nothing,
#     size=(1500, 500),
#     plot_title="Adjoint gradient w.r.t. model velocities"
# )
# savefig("adjgrad.png")

# # Plot FD gradient and zoom in
# p_grad = plot_zoom(fdgrad, corner1, plot_nice_heatmap_grad; lx=lx, ly=ly, dx=dx, dy=dy)
# p_grad_zoom = plot_zoom(
#     fdgrad[corner1:end-corner1, corner1:end-corner1],
#     corner2,
#     plot_nice_heatmap_grad;
#     lx=(nx - corner1 * 2) * dx,
#     ly=(ny - corner1 * 2) * dy,
#     dx=dx,
#     dy=dy,
#     shift=-dx * corner1
# )
# p_grad_zoom2 = plot_nice_heatmap_grad(
#     fdgrad[corner2:end-corner2, corner2:end-corner2];
#     lx=(nx - corner2 * 2) * dx,
#     ly=(ny - corner2 * 2) * dy,
#     dx=dx,
#     dy=dy,
#     shift=-dx * corner2
# )
# p = plot(
#     p_grad,
#     p_grad_zoom,
#     p_grad_zoom2;
#     layout=l,
#     legend=nothing,
#     size=(1500, 500),
#     plot_title="FD gradient w.r.t. model velocities"
# )
# savefig("fdgrad.png")

# # Plot difference between adjoint and FD grad and zoom in
# p_grad = plot_zoom(grad_diff, corner1, plot_nice_heatmap_grad; lx=lx, ly=ly, dx=dx, dy=dy)
# p_grad_zoom = plot_zoom(
#     grad_diff[corner1:end-corner1, corner1:end-corner1],
#     corner2,
#     plot_nice_heatmap_grad;
#     lx=(nx - corner1 * 2) * dx,
#     ly=(ny - corner1 * 2) * dy,
#     dx=dx,
#     dy=dy,
#     shift=-dx * corner1
# )
# p_grad_zoom2 = plot_nice_heatmap_grad(
#     grad_diff[corner2:end-corner2, corner2:end-corner2];
#     lx=(nx - corner2 * 2) * dx,
#     ly=(ny - corner2 * 2) * dy,
#     dx=dx,
#     dy=dy,
#     shift=-dx * corner2
# )
# p = plot(
#     p_grad,
#     p_grad_zoom,
#     p_grad_zoom2;
#     layout=l,
#     legend=nothing,
#     size=(1500, 500),
#     plot_title="Absolute error between adjoint and FD gradient"
# )
# savefig("grad_err.png")

# # Plot relative difference between adjoint and FD grad and zoom in
# rel_diff = log10.(abs.(grad_diff ./ fdgrad) * 100)
# p_grad = plot_zoom(rel_diff, corner1, plot_nice_heatmap; lx=lx, ly=ly, dx=dx, dy=dy)
# p_grad_zoom = plot_zoom(
#     rel_diff[corner1:end-corner1, corner1:end-corner1],
#     corner2,
#     plot_nice_heatmap;
#     lx=(nx - corner1 * 2) * dx,
#     ly=(ny - corner1 * 2) * dy,
#     dx=dx,
#     dy=dy,
#     shift=-dx * corner1
# )
# p_grad_zoom2 = plot_nice_heatmap(
#     rel_diff[corner2:end-corner2, corner2:end-corner2];
#     lx=(nx - corner2 * 2) * dx,
#     ly=(ny - corner2 * 2) * dy,
#     dx=dx,
#     dy=dy,
#     shift=-dx * corner2
# )
# p = plot(
#     p_grad,
#     p_grad_zoom,
#     p_grad_zoom2;
#     layout=l,
#     legend=nothing,
#     size=(1500, 500),
#     plot_title="Log10 of relative error % between adjoint and FD gradient"
# )
# savefig("rel_grad_err.png")
