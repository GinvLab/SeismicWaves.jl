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
        @testset "Test 1D $(parall) swmisfit!" begin
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

            # Compute misfit
            mis = swmisfit!(params, vel, [srcs => recs]; parall=parall)
            @test mis > 0
        end

        @testset "Test 2D $(parall) swmisfit!" begin
            # Physics
            c0 = 2000.0
            f0 = 10.0
            # Numerics
            nt = 1000
            nx = 200
            ny = 301
            dx = dy = dz = 10.0
            dt = sqrt(2) / (c0 * (1 / dx + 1 / dy)) / 2
            halo = 20
            rcoef = 0.0001
            params, srcs, recs, vel =
                setup_constant_vel_2D_CPML(nt, dt, nx, ny, dx, dy, c0, f0, halo, rcoef)

            # Compute misfit
            mis = swmisfit!(params, vel, [srcs => recs]; parall=parall)
            @test mis > 0
        end

        @testset "Test 3D $(parall) swmisfit!" begin
            # Physics
            c0 = 2000.0
            f0 = 10.0
            # Numerics
            nt = 1000
            nx = 101
            ny = 85
            nz = 94
            dx = dy = dz = 10.0
            dt = sqrt(3) / (c0 * (1 / dx + 1 / dy + 1 / dz)) / 3
            halo = 20
            rcoef = 0.0001
            params, srcs, recs, vel =
                setup_constant_vel_3D_CPML(nt, dt, nx, ny, nz, dx, dy, dz, c0, f0, halo, rcoef)

            # Compute misfit
            mis = swmisfit!(params, vel, [srcs => recs]; parall=parall)
            @test mis > 0
        end
    end
end
