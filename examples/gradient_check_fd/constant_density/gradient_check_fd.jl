
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

function setup(nt, c0, c0max, r, dx, dy, dt, halo, rcoef, nx, ny, parall)
    ##========================================
    # time stuff
    t = collect(Float64, range(0.0; step=dt, length=nt)) # seconds

    ##========================================
    # create constant and gaussian velocity model
    lx = (nx - 1) * dx
    ly = (ny - 1) * dy
    matprop_const = VpAcousticCDMaterialProperties(c0 .* ones(nx, ny))
    # gaussian perturbed model
    matprop_gauss = VpAcousticCDMaterialProperties(gaussian_vel_2D(nx, ny, c0, c0max, r))

    ##========================================
    # shots definition
    nshots = 10
    f0 = 10.0
    t0 = 4 / f0
    srctf = 1000.0 .* rickerstf.(t, t0, f0)
    dd = 60
    shots = linear_2D_geometry(
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
        save_file="setup.png"
    )

    ##============================================
    ## Input parameters for acoustic simulation
    boundcond = CPMLBoundaryConditionParameters(; halo=halo, rcoef=rcoef, freeboundtop=false)
    params = InputParametersAcoustic(nt, dt, (nx, ny), (dx, dy), boundcond)

    # Wave simulation builder
    wavesim = build_wavesim(params, matprop_const; smooth_radius=0, gradient=true, parall=parall, check_freq=ceil(Int, sqrt(nt)))

    return wavesim, shots, matprop_const, matprop_gauss
end

function gradient_fd_check(wavesim, shots, matprop_const, matprop_gauss; compute_fd=true)
    println("Computing forward solver")

    # compute forward gaussian
    with_logger(error_logger) do
        swforward!(wavesim, matprop_gauss, shots)
    end

    # new receivers with observed seismograms
    shots_obs = Vector{Shot}()  #Pair{ScalarSources, ScalarReceivers}}()
    for i in eachindex(shots)
        # receivers definition
        recs = ScalarReceivers(
            copy(shots[i].recs.positions),
            nt;
            observed=copy(shots[i].recs.seismograms),
            invcov=Diagonal(ones(nt))
        )
        # add pair as shot
        push!(shots_obs, Shot(; srcs=shots[i].srcs, recs=recs)) # srcs => recs)
    end

    println("Computing gradients")

    # compute gradients and misfit
    gradient, misfit = with_logger(error_logger) do
        swgradient!(wavesim, matprop_const, shots_obs; compute_misfit=true)
    end
    # save current gradient into file
    serialize("adjgrad.dat", gradient)

    println("Initial misfit: $misfit")

    if compute_fd
        # Compute gradient with finite differences
        fd_gradient = zeros(nx, ny)
        dm = -1e-3
        for i in 1:nx
            for j in 1:ny
                println("Computing ($i, $j) gradient with FD")
                vp_perturbed = copy(matprop_const.vp)
                vp_perturbed[i, j] += dm
                matprop_perturbed = VpAcousticCDMaterialProperties(vp_perturbed)
                @time new_misfit = with_logger(error_logger) do
                    swmisfit!(wavesim, matprop_perturbed, shots_obs)
                end
                println("New misfit: $new_misfit")
                fd_gradient[i, j] = (new_misfit - misfit) / dm
            end
        end
        # save FD gradient into file
        serialize("fdgrad.dat", fd_gradient)
    end
end

function load_gradients()
    # deserialize saved gradients
    grad = deserialize("adjgrad.dat")
    fdgrad = deserialize("fdgrad.dat")
    fdgrad_diff = fdgrad .- grad
    return grad, fdgrad, fdgrad_diff
end

########################################################################

# Backend selection
parall = :serial
if length(ARGS) >= 1
    if ARGS[1] == "--threads"
        parall = :threads
    elseif ARGS[1] == "--GPU"
        parall = :GPU
        devs = devices()
        if length(devs) >= 1 && length(ARGS) >= 2
            device!(parse(Int, ARGS[2]))
        end
        @show device()
    end
end

# Numerical parameters
nt = 1000
c0 = 1000
c0max = 1300
r = 35
dh = dx = dy = 5.0
dt = dh / sqrt(2) / c0max
halo = 20
rcoef = 0.0001
nx = 201
ny = 201
lx = (nx - 1) * dx
ly = (ny - 1) * dy

# setup
wavesim, shots, matprop_const, matprop_gauss = setup(nt, c0, c0max, r, dx, dy, dt, halo, rcoef, nx, ny, parall)
gradient_fd_check(wavesim, shots, matprop_const, matprop_gauss; compute_fd=true)
# load saved results
adjgrad, fdgrad, grad_diff = load_gradients()

##################################################################

# Plotting

corner1 = 21
corner2 = round(Int, 300 ÷ dx)
l = @layout([A B C])

# Plot adjoint gradient and zoom in
p_grad = plot_zoom(adjgrad, corner1, plot_nice_heatmap_grad; lx=lx, ly=ly, dx=dx, dy=dy)
p_grad_zoom = plot_zoom(
    adjgrad[corner1:end-corner1, corner1:end-corner1],
    corner2,
    plot_nice_heatmap_grad;
    lx=(nx - corner1 * 2) * dx,
    ly=(ny - corner1 * 2) * dy,
    dx=dx,
    dy=dy,
    shift=-dx * corner1
)
p_grad_zoom2 = plot_nice_heatmap_grad(
    adjgrad[corner2:end-corner2, corner2:end-corner2];
    lx=(nx - corner2 * 2) * dx,
    ly=(ny - corner2 * 2) * dy,
    dx=dx,
    dy=dy,
    shift=-dx * corner2
)
p = plot(
    p_grad,
    p_grad_zoom,
    p_grad_zoom2;
    layout=l,
    legend=nothing,
    size=(1500, 500),
    plot_title="Adjoint gradient w.r.t. model velocities"
)
savefig("adjgrad.png")

# Plot FD gradient and zoom in
p_grad = plot_zoom(fdgrad, corner1, plot_nice_heatmap_grad; lx=lx, ly=ly, dx=dx, dy=dy)
p_grad_zoom = plot_zoom(
    fdgrad[corner1:end-corner1, corner1:end-corner1],
    corner2,
    plot_nice_heatmap_grad;
    lx=(nx - corner1 * 2) * dx,
    ly=(ny - corner1 * 2) * dy,
    dx=dx,
    dy=dy,
    shift=-dx * corner1
)
p_grad_zoom2 = plot_nice_heatmap_grad(
    fdgrad[corner2:end-corner2, corner2:end-corner2];
    lx=(nx - corner2 * 2) * dx,
    ly=(ny - corner2 * 2) * dy,
    dx=dx,
    dy=dy,
    shift=-dx * corner2
)
p = plot(
    p_grad,
    p_grad_zoom,
    p_grad_zoom2;
    layout=l,
    legend=nothing,
    size=(1500, 500),
    plot_title="FD gradient w.r.t. model velocities"
)
savefig("fdgrad.png")

# Plot difference between adjoint and FD grad and zoom in
p_grad = plot_zoom(grad_diff, corner1, plot_nice_heatmap_grad; lx=lx, ly=ly, dx=dx, dy=dy)
p_grad_zoom = plot_zoom(
    grad_diff[corner1:end-corner1, corner1:end-corner1],
    corner2,
    plot_nice_heatmap_grad;
    lx=(nx - corner1 * 2) * dx,
    ly=(ny - corner1 * 2) * dy,
    dx=dx,
    dy=dy,
    shift=-dx * corner1
)
p_grad_zoom2 = plot_nice_heatmap_grad(
    grad_diff[corner2:end-corner2, corner2:end-corner2];
    lx=(nx - corner2 * 2) * dx,
    ly=(ny - corner2 * 2) * dy,
    dx=dx,
    dy=dy,
    shift=-dx * corner2
)
p = plot(
    p_grad,
    p_grad_zoom,
    p_grad_zoom2;
    layout=l,
    legend=nothing,
    size=(1500, 500),
    plot_title="Absolute error between adjoint and FD gradient"
)
savefig("grad_err.png")

# Plot relative difference between adjoint and FD grad and zoom in
rel_diff = log10.(abs.(grad_diff ./ fdgrad) * 100)
p_grad = plot_zoom(rel_diff, corner1, plot_nice_heatmap; lx=lx, ly=ly, dx=dx, dy=dy)
p_grad_zoom = plot_zoom(
    rel_diff[corner1:end-corner1, corner1:end-corner1],
    corner2,
    plot_nice_heatmap;
    lx=(nx - corner1 * 2) * dx,
    ly=(ny - corner1 * 2) * dy,
    dx=dx,
    dy=dy,
    shift=-dx * corner1
)
p_grad_zoom2 = plot_nice_heatmap(
    rel_diff[corner2:end-corner2, corner2:end-corner2];
    lx=(nx - corner2 * 2) * dx,
    ly=(ny - corner2 * 2) * dy,
    dx=dx,
    dy=dy,
    shift=-dx * corner2
)
p = plot(
    p_grad,
    p_grad_zoom,
    p_grad_zoom2;
    layout=l,
    legend=nothing,
    size=(1500, 500),
    plot_title="Log10 of relative error % between adjoint and FD gradient"
)
savefig("rel_grad_err.png")
