using Test

using SeismicWaves
import SeismicWaves.Acoustic1D_CUDA
import SeismicWaves.Acoustic2D_CUDA
import SeismicWaves.Acoustic3D_CUDA

using LinearAlgebra

using Logging
error_logger = ConsoleLogger(stderr, Logging.Error)
with_logger(error_logger) do
    @testset "Test 1D gradient! checkpointing" begin
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
        grad = gradient!(model, [srcs => recs], SeismicWaves.Acoustic1D_CUDA; check_freq=nothing)
        # Solve gradient with (optimal) checkpointing
        grad_check = gradient!(model, [srcs => recs], SeismicWaves.Acoustic1D_CUDA; check_freq=floor(Int, sqrt(model.nt)))

        # Check that gradient is non zero
        @test !all(g -> g == 0.0, grad)
        # Check that computations are equivalent
        @test grad ≈ grad_check
    end

    @testset "Test 2D gradient! checkpointing" begin
        # Physics
        c0 = 2000.0
        # Numerics
        nt = 1000
        nx = ny = 101
        dx = dy = 10.0
        dt = sqrt(2) / (c0 * (1/dx + 1/dy)) / 2
        vel = c0 .* ones(nx, ny)
        halo = 20
        rcoef = 0.0001
        model = IsotropicAcousticCPMLWaveModel2D(nt, dt, dx, dy, halo, rcoef, vel)
        # Sources
        f0 = 10.0
        t0 = 4 / f0
        times = collect(range(0.0, step=dt, length=nt))
        possrcs = zeros(1,2)
        possrcs[1,:] = [model.lx / 2, model.ly / 2]
        srctf = zeros(nt, 1)
        srctf[:,1] .= rickersource1D.(times, t0, f0)
        srcs = Sources(possrcs, srctf, f0)
        # Receivers
        posrecs = zeros(1,2)
        posrecs[1,:] = [model.lx / 4, model.ly / 2]
        observed = copy(srctf)
        invcov = Diagonal(ones(nt))
        recs = Receivers(posrecs, nt; observed=observed, invcov=invcov)

        # Solve gradient without checkpointing
        grad = gradient!(model, [srcs => recs], SeismicWaves.Acoustic2D_CUDA; check_freq=nothing)
        # Solve gradient with (optimal) checkpointing
        grad_check = gradient!(model, [srcs => recs], SeismicWaves.Acoustic2D_CUDA; check_freq=floor(Int, sqrt(model.nt)))

        # Check that gradient is non zero
        @test !all(g -> g == 0.0, grad)
        # Check that computations are equivalent
        @test grad ≈ grad_check
    end

    @testset "Test 3D gradient! checkpointing" begin
        # Physics
        c0 = 2000.0
        # Numerics
        nt = 100
        nx = ny = nz = 101
        dx = dy = dz = 10.0
        dt = sqrt(3) / (c0 * (1/dx + 1/dy + 1/dz)) / 3
        vel = c0 .* ones(nx, ny, nz)
        halo = 20
        rcoef = 0.0001
        model = IsotropicAcousticCPMLWaveModel3D(nt, dt, dx, dy, dz, halo, rcoef, vel)
        # Sources
        f0 = 10.0
        t0 = 4 / f0
        times = collect(range(0.0, step=dt, length=nt))
        possrcs = zeros(1,3)
        possrcs[1,:] = [model.lx / 2, model.ly / 2, model.lz / 2]
        srctf = zeros(nt, 1)
        srctf[:,1] .= rickersource1D.(times, t0, f0)
        srcs = Sources(possrcs, srctf, f0)
        # Receivers
        posrecs = zeros(1,3)
        posrecs[1,:] = [model.lx / 2 + 100, model.ly / 2, model.lz / 2]
        observed = copy(srctf)
        invcov = Diagonal(ones(nt))
        recs = Receivers(posrecs, nt; observed=observed, invcov=invcov)

        # Solve gradient without checkpointing
        grad = gradient!(model, [srcs => recs], SeismicWaves.Acoustic3D_CUDA; check_freq=nothing)
        # Solve gradient with (optimal) checkpointing
        grad_check = gradient!(model, [srcs => recs], SeismicWaves.Acoustic3D_CUDA; check_freq=floor(Int, sqrt(model.nt)))

        # Check that gradient is non zero
        @test !all(g -> g == 0.0, grad)
        # Check that computations are equivalent
        @test grad ≈ grad_check
    end
end