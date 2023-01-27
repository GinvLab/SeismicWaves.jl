using Revise, LinearAlgebra, BenchmarkTools, Plots
using Waves
import Waves.Acoustic2D_Threads

using Logging
debug_logger = ConsoleLogger(stderr, Logging.Debug)
global_logger(debug_logger)

c0 = 2000.0
nt = 1000
nx = 101
ny = 101
# nz = 101
dt = 0.001 / 4
dx = 10.0
dy = 10.0
# dz = 10.0
vel = c0 .* ones(nx, ny)

halo = 20
rcoef = 0.0001

f0 = 10.0
t0 = 4 / f0
times = collect(range(0.0, step=dt, length=nt))

model = IsotropicAcousticCPMLWaveModel2D(nt, dt, dx, dy, halo, rcoef, vel)
possrcs = zeros(1,2)
possrcs[1,:] = [model.lx / 2, model.ly / 2]
posrecs = zeros(1,2)
posrecs[1,:] = [model.lx / 4, model.ly / 4]
srctf = zeros(nt, 1)
srctf[:,1] .= rickersource1D.(times, t0, f0)
observed = copy(srctf)
invcov = Diagonal(ones(nt))

srcs = Sources(possrcs, srctf, f0)
recs = Receivers(posrecs, nt; observed=observed, invcov=invcov)

foo() = solve_gradient!(model, [srcs => recs], Waves.Acoustic2D_Threads; check_freq=nothing)
bar() = solve_gradient!(model, [srcs => recs], Waves.Acoustic2D_Threads; check_freq=floor(Int, sqrt(model.nt)))

grad = foo()
grad2 = bar()

# b1 = @benchmark foo()
# b2 = @benchmark bar()

@show norm(grad .- grad2)

# @show b1
# @show b2