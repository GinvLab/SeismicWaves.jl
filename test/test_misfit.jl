using Test

using SeismicWaves
import SeismicWaves.Acoustic1D_Threads

using LinearAlgebra

using Logging
error_logger = ConsoleLogger(stderr, Logging.Error)
with_logger(error_logger) do
    @testset "Test 1D misfit!" begin
        # Physics
        c0 = 2000.0
        # Numerics
        nt = 1000
        nx = 101
        dx = 10.0
        dt = dx / c0
        vel = c0 .* ones(nx)
        halo = 20
        rcoef = 0.0001
        model = IsotropicAcousticCPMLWaveModel1D(nt, dt, dx, halo, rcoef, vel)
        # Sources
        f0 = 10.0
        t0 = 4 / f0
        times = collect(range(0.0, step=dt, length=nt))
        possrcs = zeros(1,1)
        possrcs[1,:] = [model.lx / 2]
        srctf = zeros(nt, 1)
        srctf[:,1] .= rickersource1D.(times, t0, f0)
        srcs = Sources(possrcs, srctf, f0)
        # Receivers
        posrecs = zeros(1,1)
        posrecs[1,:] = [model.lx / 4]
        observed = copy(srctf)
        invcov = Diagonal(ones(nt))
        recs = Receivers(posrecs, nt; observed=observed, invcov=invcov)

        # Solve gradient without checkpointing
        mis = misfit!(model, [srcs => recs], SeismicWaves.Acoustic1D_Threads)
        @test mis > 0
    end
end