using Plots, Revise
using Waves
import Waves.Acoustic1D

using Logging
debug_logger = ConsoleLogger(stderr, Logging.Debug)
global_logger(debug_logger)

nt = 10000
nx = 1001
dt = 0.01
dx = 0.01
vel = ones(nx)

halo = 20
rcoef = 0.0001

f0 = 1.0
t0 = 4 / f0
times = collect(range(0.0, step=dt, length=nt))

model = IsotropicAcousticCPMLWaveModel1D(nt, dt, dx, halo, rcoef, vel, 100)
possrcs = zeros(1,1)
possrcs[1,1] = 5.0
posrecs = zeros(1,1)
posrecs[1,1] = 2.0
srctf = zeros(nt, 1)
srctf[:,1] .= rickersource1D.(times, t0, f0)

srcs = Sources(possrcs, srctf, f0)
recs = Receivers(posrecs, nt)

res = solve!(model, [srcs => recs], Waves.Acoustic1D)

display(plot(times, recs.seismograms[:,1]))