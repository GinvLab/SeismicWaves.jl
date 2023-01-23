using Plots, Revise
using Waves
import Waves.Acoustic3D_Threads

using Logging
debug_logger = ConsoleLogger(stderr, Logging.Debug)
global_logger(debug_logger)

nt = 4000
nx = 101
ny = 101
nz = 101
dt = 0.0014 / 3
dx = 10.0
dy = 10.0
dz = 10.0
vel = 2000.0 .* ones(nx,ny,nz)

halo = 20
rcoef = 0.0001

f0 = 4.0
t0 = 4 / f0
times = collect(range(0.0, step=dt, length=nt))

model = IsotropicAcousticCPMLWaveModel3D(nt, dt, dx, dy, dz, halo, rcoef, vel, false, 500)
possrcs = zeros(1,3)
possrcs[1,:] = [model.lx / 2, model.ly / 2, model.lz / 2]
posrecs = zeros(1,3)
posrecs[1,:] = [model.lx / 2, model.ly / 4, model.lz / 2]
srctf = zeros(nt, 1)
srctf[:,1] .= rickersource1D.(times, t0, f0)

srcs = Sources(possrcs, srctf, f0)
recs = Receivers(posrecs, nt)

res = solve!(model, [srcs => recs], Waves.Acoustic3D_Threads)

display(plot(times, recs.seismograms[:,1]))
