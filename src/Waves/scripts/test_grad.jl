using Plots, Revise, LinearAlgebra
using Waves
import Waves.Acoustic1D_Threads

using Logging
debug_logger = ConsoleLogger(stderr, Logging.Debug)
global_logger(debug_logger)

nt = 13
nx = 101
# ny = 101
# nz = 101
dt = 0.001
dx = 10.0
# dy = 10.0
# dz = 10.0
vel = 2000.0 .* ones(nx)

halo = 20
rcoef = 0.0001

f0 = 4.0
t0 = 4 / f0
times = collect(range(0.0, step=dt, length=nt))

model = IsotropicAcousticCPMLWaveModel1D(nt, dt, dx, halo, rcoef, vel)
possrcs = zeros(1,1)
possrcs[1,:] = [model.lx / 2]
posrecs = zeros(1,1)
posrecs[1,:] = [model.lx / 4]
srctf = zeros(nt, 1)
srctf[:,1] .= rickersource1D.(times, t0, f0)

srcs = Sources(possrcs, srctf, f0)
recs = Receivers(posrecs, nt, copy(srctf))
invcov = Diagonal(ones(nt))

solve_gradient!(model, [srcs => recs], invcov, Waves.Acoustic1D_Threads)
