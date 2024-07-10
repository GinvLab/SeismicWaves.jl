using Test
using DSP, NumericalIntegration, LinearAlgebra
using CUDA: CUDA
using Logging
using SeismicWaves

with_logger(ConsoleLogger(stderr, Logging.Warn)) do
    test_backends = [:threads]
    # test GPU backend only if CUDA is functional
    if @isdefined(CUDA) && CUDA.functional()
        push!(test_backends, :CUDA)
    end
    if @isdefined(AMDGPU) && AMDGPU.functional()
            push!(test_backends, :AMDGPU)
    end

    for parall in test_backends
        @testset "Test 1D $(parall) swgradient! with compute misfit" begin
            # Physics
            c0 = 2000.0
            ρ0 = 1500.0
            f0 = 10.0
            t0 = 2 / f0
            # Numerics
            nt = 1000
            nx = 101
            dx = 10.0
            dt = dx / c0 * 6 / 7
            halo = 20
            rcoef = 0.0001
            params, shots, matprop = setup_constant_vel_rho_1D_CPML(nt, dt, nx, dx, c0, ρ0, t0, f0, halo, rcoef)

            # Compute gradient and misfit
            grad, misfit = swgradient!(
                params,
                matprop,
                shots;
                parall=parall,
                check_freq=nothing,
                compute_misfit=true
            )
            # Compute only misfit
            misfit_check = swmisfit!(params, matprop, shots; parall=parall)

            # Check that gradient is non zero
            @test !all(g -> g == 0.0, grad["vp"])
            @test !all(g -> g == 0.0, grad["rho"])
            # Check that misfits are non zero
            @test !(misfit ≈ 0.0)
            @test !(misfit_check ≈ 0.0)
            # Check that misfits are equivalent
            @test misfit ≈ misfit_check
        end

        @testset "Test 2D $(parall) swgradient! with compute misfit" begin
            # Physics
            c0 = 2000.0
            ρ0 = 1500.0
            f0 = 10.0
            t0 = 2 / f0
            # Numerics
            nt = 1000
            nx = ny = 101
            dx = dy = 10.0
            dt = dx / c0 / sqrt(2) * 6 / 7
            halo = 20
            rcoef = 0.0001
            params, shots, matprop = setup_constant_vel_rho_2D_CPML(nt, dt, nx, ny, dx, dy, c0, ρ0, t0, f0, halo, rcoef)

            # Compute gradient and misfit
            grad, misfit = swgradient!(
                params,
                matprop,
                shots;
                parall=parall,
                check_freq=nothing,
                compute_misfit=true
            )
            # Compute only misfit
            misfit_check = swmisfit!(params, matprop, shots; parall=parall)

            # Check that gradient is non zero
            @test !all(g -> g == 0.0, grad["vp"])
            @test !all(g -> g == 0.0, grad["rho"])
            # Check that misfits are non zero
            @test !(misfit ≈ 0.0)
            @test !(misfit_check ≈ 0.0)
            # Check that misfits are equivalent
            @test misfit ≈ misfit_check
        end

        @testset "Test 1D $(parall) swgradient! checkpointing" begin
            # Physics
            c0 = 2000.0
            ρ0 = 1500.0
            f0 = 10.0
            t0 = 2 / f0
            # Numerics
            nt = 1000
            nx = 101
            dx = 10.0
            dt = dx / c0 * 6 / 7
            halo = 20
            rcoef = 0.0001
            params, shots, matprop = setup_constant_vel_rho_1D_CPML(nt, dt, nx, dx, c0, ρ0, t0, f0, halo, rcoef)

            # Solve gradient without checkpointing
            grad = swgradient!(
                params,
                matprop,
                shots;
                parall=parall,
                check_freq=nothing
            )
            # Solve gradient with (optimal) checkpointing
            grad_check = swgradient!(
                params,
                matprop,
                shots;
                parall=parall,
                check_freq=floor(Int, sqrt(nt))
            )

            # Check that gradient is non zero
            @test !all(g -> g == 0.0, grad["vp"])
            @test !all(g -> g == 0.0, grad["rho"])
            # Check that computations are equivalent
            @test grad["vp"] ≈ grad_check["vp"]
            @test grad["rho"] ≈ grad_check["rho"]
        end

        @testset "Test 2D $(parall) swgradient! checkpointing" begin
            # Physics
            c0 = 2000.0
            ρ0 = 1500.0
            f0 = 10.0
            t0 = 2 / f0
            # Numerics
            nt = 1000
            nx = ny = 101
            dx = dy = 10.0
            dt = dx / c0 / sqrt(2) * 6 / 7
            halo = 20
            rcoef = 0.0001
            params, shots, matprop = setup_constant_vel_rho_2D_CPML(nt, dt, nx, ny, dx, dy, c0, ρ0, t0, f0, halo, rcoef)

            # Solve gradient without checkpointing
            grad = swgradient!(
                params,
                matprop,
                shots;
                parall=parall,
                check_freq=nothing
            )
            # Solve gradient with (optimal) checkpointing
            grad_check = swgradient!(
                params,
                matprop,
                shots;
                parall=parall,
                check_freq=floor(Int, sqrt(nt))
            )

            # Check that gradient is non zero
            @test !all(g -> g == 0.0, grad["vp"])
            @test !all(g -> g == 0.0, grad["rho"])
            # Check that computations are equivalent
            @test grad["vp"] ≈ grad_check["vp"]
            @test grad["rho"] ≈ grad_check["rho"]
        end
    end
end
