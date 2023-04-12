using Test
using SeismicWaves
using CUDA: CUDA

include("utils.jl")

using Logging
error_logger = ConsoleLogger(stderr, Logging.Error)
with_logger(error_logger) do
    test_backends = [:serial, :threads]
    # test GPU backend only if CUDA is functional
    if CUDA.functional()
        push!(test_backends, :GPU)
    end

    for parall in test_backends
        @testset "Test 1D $(parall) swgradient! checkpointing" begin
            # Physics
            c0 = 2000.0
            f0 = 10.0
            # Numerics
            nt = 1000
            nx = 101
            dx = 10.0
            dt = dx / c0
            halo = 20
            rcoef = 0.0001
            params, srcs, recs, vel =
                setup_constant_vel_1D_CPML(nt, dt, nx, dx, c0, f0, halo, rcoef)

            # Solve gradient without checkpointing
            grad = swgradient!(params,
                vel,
                [srcs => recs];
                parall=parall,
                check_freq=nothing)
            # Solve gradient with (optimal) checkpointing
            grad_check = swgradient!(params,
                vel,
                [srcs => recs];
                parall=parall,
                check_freq=floor(Int, sqrt(nt)))

            # Check that gradient is non zero
            @test !all(g -> g == 0.0, grad)
            # Check that computations are equivalent
            @test grad ≈ grad_check
        end

        @testset "Test 2D $(parall) swgradient! checkpointing" begin
            # Physics
            c0 = 2000.0
            f0 = 10.0
            # Numerics
            nt = 1000
            nx = ny = 101
            dx = dy = 10.0
            dt = sqrt(2) / (c0 * (1 / dx + 1 / dy)) / 2
            halo = 20
            rcoef = 0.0001
            params, srcs, recs, vel =
                setup_constant_vel_2D_CPML(nt, dt, nx, ny, dx, dy, c0, f0, halo, rcoef)

            # Solve gradient without checkpointing
            grad = swgradient!(params,
                vel,
                [srcs => recs];
                parall=parall,
                check_freq=nothing)
            # Solve gradient with (optimal) checkpointing
            grad_check = swgradient!(params,
                vel,
                [srcs => recs];
                parall=parall,
                check_freq=floor(Int, sqrt(nt)))

            # Check that gradient is non zero
            @test !all(g -> g == 0.0, grad)
            # Check that computations are equivalent
            @test grad ≈ grad_check
        end

        @testset "Test 3D $(parall) swgradient! checkpointing" begin
            # Physics
            c0 = 2000.0
            f0 = 10.0
            # Numerics
            nt = 100
            nx = ny = nz = 81
            dx = dy = dz = 10.0
            dt = sqrt(3) / (c0 * (1 / dx + 1 / dy + 1 / dz)) / 3
            halo = 20
            rcoef = 0.0001
            params, srcs, recs, vel =
                setup_constant_vel_3D_CPML(nt, dt, nx, ny, nz, dx, dy, dz, c0, f0, halo, rcoef)

            # Solve gradient without checkpointing
            grad = swgradient!(params,
                vel,
                [srcs => recs];
                parall=parall,
                check_freq=nothing)
            # Solve gradient with (optimal) checkpointing
            grad_check = swgradient!(params,
                vel,
                [srcs => recs];
                parall=parall,
                check_freq=floor(Int, sqrt(nt)))

            # Check that gradient is non zero
            @test !all(g -> g == 0.0, grad)
            # Check that computations are equivalent
            @test grad ≈ grad_check
        end
    end
end
