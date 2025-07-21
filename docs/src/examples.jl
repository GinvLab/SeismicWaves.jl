# # [Examples](@id examples)

# ## Acoustic wave simulation example

using SeismicWaves

## numerics
nt = 1500
dt = 0.0012
dh = 8.0
t = collect(Float64, range(0.0; step=dt, length=nt)) ## time vector

## create a velocity model (gradient from top to bottom)
nx = 211
nz = 120
velmod = zeros(nx, nz)
for i in 1:nx
    for j in 1:nz
        velmod[i, j] = 2000.0 + 12.0 * (j - 1)
    end
end
matprop = VpAcousticCDMaterialProperties(velmod)

## shots definition
nshots = 6
shots = Vector{ScalarShot{Float64}}()
## sources x-position (in grid points) (different for every shot)
ixsrc = round.(Int, LinRange(32, nx - 31, nshots))
for i in 1:nshots
    ## sources definition
    nsrc = 1
    possrcs = zeros(1, 2)                ## 1 source, 2 dimensions
    possrcs[:, 1] .= (ixsrc[i] - 1) * dh    ## x-positions in meters
    possrcs[:, 2] .= 2 * dh                 ## y-positions in meters
    ## source time functions
    srcstf = zeros(nt, nsrc)
    for s in 1:nsrc
        srcstf[:, s] .= 1000.0 .* rickerstf.(t, 1.20 / 12.0, 12.0)
    end
    srcs = ScalarSources(possrcs, srcstf, 12.0)

    ## receivers definition
    nrecs = 20
    ## receivers x-positions (in grid points) (same for every shot)
    ixrec = round.(Int, LinRange(30, nx - 29, nrecs))
    posrecs = zeros(nrecs, 2)              ## 20 receivers, 2 dimensions
    posrecs[:, 1] .= (ixrec .- 1) .* dh    ## x-positions in meters
    posrecs[:, 2] .= 2 * dh                ## y-positions in meters
    recs = ScalarReceivers(posrecs, nt)

    ## add pair as shot
    push!(shots, ScalarShot(; srcs=srcs, recs=recs)) ## srcs => recs)
end

## Input parameters for acoustic simulation
snapevery = 100
boundcond = CPMLBoundaryConditionParameters(; halo=20, rcoef=0.0001, freeboundtop=true)
params = InputParametersAcoustic(nt, dt, (nx, nz), (dh, dh), boundcond)

## Compute the seismograms
runparams = RunParameters(parall=:threads,snapevery=snapevery)

snapshots = swforward!(
    params,
    matprop,
    shots;
    runparams=runparams
)

# ## Elastic wave simulation example

## time stuff
nt = 3000 #1500
dt = 0.0008
t = collect(Float64, range(0.0; step=dt, length=nt)) # seconds

## create a velocity model
nx = 380 ## 211
nz = 270 ## 120
dh = 4.5 ## meters

vp = zeros(nx, nz)
vp[nx÷2+10:end, :] .= 3100.0
for i in 1:nx
    for j in 1:nz
        vp[i, j] = 2000.0 + dh * (j - 1)
    end
end
vs = vp ./ sqrt(3)

ρ = 2100.0 * ones(nx, nz)
μ = vs .^ 2 .* ρ  ## μ = Vs^2⋅ρ 
λ = (vp .^ 2 .* ρ) .- (2 .* μ)  ## λ = vp^2 · ρ - 2μ

matprop = ElasticIsoMaterialProperties(; λ=λ, μ=μ, ρ=ρ)

## shots definition
nshots = 1
shots = Vector{MomentTensorShot{Float64, 2, MomentTensor2D{Float64}}}()

for i in 1:nshots
    ## sources definition
    nsrc = 1
    ## sources x-position (in grid points) (different for every shot)
    if nsrc == 1
        ixsrc = [nx / 2]
    else
        ixsrc = round.(Int, LinRange(30, nx - 31, nsrc))
    end
    possrcs = zeros(nsrc, 2)    ## 1 source, 2 dimensions
    for s in 1:nsrc
        possrcs[s, 1] = (ixsrc[i] - 1) * dh .+ 0.124   ## x-positions in meters
        possrcs[s, 2] = (nz / 2) * dh .+ 0.124 ## y-positions in meters
    end

    ## source time functions
    f0 = 12.0
    t0 = 1.20 / f0
    srcstf = zeros(nt, nsrc)
    Mxx = zeros(nsrc)
    Mzz = zeros(nsrc)
    Mxz = zeros(nsrc)
    for s in 1:nsrc
        srcstf[:, s] .= rickerstf.(t, t0, f0)
        Mxx[s] = 5e10
        Mzz[s] = 5e10
        Mxz[s] = 0.89e10
    end

    srcs = MomentTensorSources(
        possrcs, srcstf,
        [MomentTensor2D(; Mxx=Mxx[s], Mzz=Mzz[s], Mxz=Mxz[s]) for s in 1:nsrc],
        f0
    )

    ## receivers definition
    nrecs = 10
    ## receivers x-positions (in grid points) (same for every shot)
    ixrec = round.(Int, LinRange(40, nx - 40, nrecs))
    posrecs = zeros(nrecs, 2)    ## 20 receivers, 2 dimensions
    posrecs[:, 1] .= (ixrec .- 1) .* dh .- 0.324   ## x-positions in meters
    posrecs[:, 2] .= 3 * dh                        ## y-positions in meters

    ndim = 2
    recs = VectorReceivers(posrecs, nt, ndim)

    ## add pair as shot
    push!(shots, MomentTensorShot(; srcs=srcs, recs=recs)) # srcs => recs)
end

## Input parameters for elastic simulation
snapevery = 5
infoevery = 100
freetop = true
halo = 20
rcoef = 0.0001
boundcond = CPMLBoundaryConditionParameters(; halo=halo, rcoef=rcoef, freeboundtop=freetop)
params = InputParametersElastic(nt, dt, (nx, nz), (dh, dh), boundcond)

## Compute the seismograms
runparams = RunParameters(parall=:threads,infoevery=infoevery,snapevery=snapevery)

snapshots = swforward!(
    params,
    matprop,
    shots;
    runparams=runparams
)
