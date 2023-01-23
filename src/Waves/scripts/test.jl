using Plots, Revise
using Waves
import Waves.Acoustic2D_Threads

using Logging
debug_logger = ConsoleLogger(stderr, Logging.Debug)
global_logger(debug_logger)

nt = 2000
nx = 101
ny = 101
dt = 0.0014
dx = 10.0
dy = 10.0
vel = 2000.0 .* ones(nx,ny)

halo = 20
rcoef = 0.0001

f0 = 8.0
t0 = 4 / f0
times = collect(range(0.0, step=dt, length=nt))

model = IsotropicAcousticCPMLWaveModel2D(nt, dt, dx, dy, halo, rcoef, vel, false, 5)
possrcs = zeros(1,2)
possrcs[1,:] = [model.lx / 2, model.ly / 2]
posrecs = zeros(1,2)
posrecs[1,:] = [model.lx / 2, model.ly / 4]
srctf = zeros(nt, 1)
srctf[:,1] .= rickersource1D.(times, t0, f0)

srcs = Sources(possrcs, srctf, f0)
recs = Receivers(posrecs, nt)

res = solve!(model, [srcs => recs], Waves.Acoustic2D_Threads)

display(plot(times, recs.seismograms[:,1]))