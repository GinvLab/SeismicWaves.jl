
using SeismicWaves
using LinearAlgebra

###################################################################
using Logging
using Plots

error_logger = ConsoleLogger(stderr, Logging.Debug)

function gaussian_vel_1D(nx, c0, c0max, r, origin=(nx+1)/2)
    sigma = r / 3
    amp = c0max - c0
    f(x) = amp * exp(-(0.5 * (x - origin)^2 / sigma^2))
    return c0 .+ [f(x) for x in 1:nx]
end

function gaussian_vel_2D(nx, ny, c0, c0max, r, origin=[(nx+1)/2, (ny+1)/2])
    sigma = r / 3
    amp = c0max - c0
    f(x,y) = amp * exp(-(0.5 * ((x - origin[1])^2 + (y - origin[2])^2) / sigma^2))
    return c0 .+ [f(x,y) for x in 1:nx, y in 1:ny]
end

function gaussian_vel_3D(nx, ny, nz, c0, c0max, r, origin=[(nx+1)/2, (ny+1)/2, (nz+1)/2])
    sigma = r / 3
    amp = c0max - c0
    f(x,y,z) = amp * exp(-(0.5 * ((x - origin[1])^2 + (y - origin[2])^2 + (z - origin[3])^2) / sigma^2))
    return c0 .+ [f(x,y,z) for x in 1:nx, y in 1:ny, z in 1:nz]
end

##========================================
# time stuff
nt = 1000
c0 = 1000
c0max = 1300
r = 250
dh = 5.0
dt = dh / sqrt(2) / c0max
t = collect(Float64, range(0.0; step=dt, length=nt)) # seconds

##========================================
# create a velocity model
nx = 201
nz = 201
matprop_const = VpAcousticCDMaterialProperty(c0 .* ones(nx, nz))
# gaussian perturbed model
matprop_gauss = VpAcousticCDMaterialProperty(gaussian_vel_2D(nx, nz, c0, c0max, r))

##========================================
# shots definition
nshots = 1
shots = Vector{Shot{Float64}}()  #Pair{Sources, Receivers}}()
# sources and receivers positions (in meters)
ixs = LinRange(100, 900, 9)
izsrc = 100
izrec = 900
# source time function
f0 = 10
t0 = 4 / f0
srcstf = reshape(1000.0 .* rickersource1D.(t, t0, f0), nt, 1)

for i in 1:nshots
    # sources definition
    possrcs = [ixs[i] izsrc]
    srcs = Sources(possrcs, copy(srcstf), f0)

    # receivers definition
    nrecs = 9
    posrecs = zeros(nrecs, 2)
    posrecs[:, 1] .= reverse(ixs)         # x-positions in meters
    posrecs[:, 2] .= izrec                # y-positions in meters
    recs = Receivers(posrecs, nt)

    # add pair as shot
    push!(shots, Shot(; srcs=srcs, recs=recs)) # srcs => recs)
end

##============================================
## Input parameters for acoustic simulation
boundcond = CPMLBoundaryConditionParameters(; halo=40, rcoef=0.00001, freeboundtop=false)
params = InputParametersAcoustic(nt, dt, [nx, nz], [dh, dh], boundcond)

# Wave simulation builder
wavesim = build_wavesim(params; gradient=true, parall=:threads, infoevery=250, check_freq=ceil(Int, sqrt(nt)))

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

# compute gradients and misfit
gradient, misfit = with_logger(error_logger) do
    swgradient!(wavesim, matprop_const, shots_obs; compute_misfit=true)
end

##################################################################
