
using SeismicWaves
using LinearAlgebra

###################################################################
using Logging
using Plots

using CUDA

info_logger = ConsoleLogger(stderr, Logging.Info)
warn_logger = ConsoleLogger(stderr, Logging.Warn)
error_logger = ConsoleLogger(stderr, Logging.Error)
debug_logger = ConsoleLogger(stderr, Logging.Debug)

include("models.jl")
include("geometries.jl")
include("plotting_utils.jl")

##========================================
# backend selection
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

##========================================
# time stuff
nt = 1000
c0 = 1000
c0max = 1300
r = 35
dh = dx = dy = 5.0
dt = dh / sqrt(2) / c0max
halo = 20
rcoef = 0.0001
t = collect(Float64, range(0.0; step=dt, length=nt)) # seconds

##========================================
# create constant and gaussian velocity model
nx = 201
ny = 201
lx = (nx-1) * dx
ly = (ny-1) * dy
matprop_const = VpAcousticCDMaterialProperty(c0 .* ones(nx, ny))
# gaussian perturbed model
matprop_gauss = VpAcousticCDMaterialProperty(gaussian_vel_2D(nx, ny, c0, c0max, r))

##========================================
# shots definition
nshots = 10
f0 = 10
t0 = 4 / f0
srctf = 1000.0 .* rickersource1D.(t, t0, f0)
dd = 60
shots = linear_2D_geometry(nshots, matprop_gauss.vp, f0, nt, srctf, dd, lx, ly, dx, dy, halo; plot_geometry=true, save_file="tmp.png")

##============================================
## Input parameters for acoustic simulation
boundcond = CPMLBoundaryConditionParameters(; halo=halo, rcoef=rcoef, freeboundtop=false)
params = InputParametersAcoustic(nt, dt, [nx, ny], [dx, dy], boundcond)

# Wave simulation builder
wavesim = build_wavesim(params; gradient=true, parall=parall, check_freq=ceil(Int, sqrt(nt)))

println("Computing forward solver")

# compute forward gaussian
with_logger(error_logger) do
    swforward!(wavesim, matprop_gauss, shots)
end

# new receivers with observed seismograms
shots_obs = Vector{Shot{Float64}}()  #Pair{Sources, Receivers}}()
for i in 1:nshots
    # receivers definition
    recs = Receivers(copy(shots[i].recs.positions), nt; observed=copy(shots[i].recs.seismograms), invcov=Diagonal(ones(nt)))
    # add pair as shot
    push!(shots_obs, Shot(; srcs=shots[i].srcs, recs=recs)) # srcs => recs)
end

println("Computing gradients")

# compute gradients and misfit
gradient, misfit = with_logger(error_logger) do
    swgradient!(wavesim, matprop_const, shots_obs; compute_misfit=true)
end

println("Initial misfit: $misfit")
plot_nice_heatmap_grad(gradient; lx=lx, ly=ly, dx=dx, dy=dy)
savefig("grad.png")

# Compute gradient with finite differences
fd_gradient = zeros(nx, ny)
dm = 1e-3
for i in 1:nx
    for j in 1:ny
        println("Computing ($i, $j) gradient with FD")
        vp_perturbed = copy(matprop_const.vp)
        vp_perturbed[i,j] += dm
        matprop_perturbed = VpAcousticCDMaterialProperty(vp_perturbed)
        @time new_misfit = with_logger(error_logger) do
            swmisfit!(wavesim, matprop_perturbed, shots_obs)
        end
        println("New misfit: $new_misfit")
        fd_gradient[i,j] = (new_misfit - misfit) / dm
    end
end

plot_nice_heatmap_grad(fd_gradient; lx=lx, ly=ly, dx=dx, dy=dy)
savefig("fdgrad.png")

plot_nice_heatmap_grad(fd_gradient .- gradient; lx=lx, ly=ly, dx=dx, dy=dy)
savefig("grad_diff.png")

##################################################################
