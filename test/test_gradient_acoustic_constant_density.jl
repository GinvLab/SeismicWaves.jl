
with_logger(ConsoleLogger(stderr, Logging.Warn)) do
    test_backends = [:serial, :threads]
    # test GPU backend only if CUDA is functional
    if @isdefined(CUDA) && CUDA.functional()
        push!(test_backends, :CUDA)
    end
    if @isdefined(AMDGPU) && AMDGPU.functional()
        push!(test_backends, :AMDGPU)
    end

    @testset "Test gradient (acoustic CD)" begin

    for parall in test_backends
        # parallelisation
        runparams = RunParameters(parall=parall)

        @testset "Test 1D $(parall) swgradient! with compute misfit" begin
            # Physics
            c0 = 2000.0
            f0 = 10.0
            # Numerics
            nt = 1000
            nx = 101
            dx = 10.0
            dt = 0.99 * dx / c0
            halo = 20
            rcoef = 0.0001
            params, shots, misfitobj, vel = setup_constant_vel_1D_CPML(nt, dt, nx, dx, c0, f0, halo, rcoef)

            # Compute gradient and misfit
            gradparams = GradParameters(compute_misfit=true)
            grad, misfit = swgradient!(
                params,
                vel,
                shots,
                misfitobj;
                runparams=runparams,
                gradparams=gradparams
            )
            # Compute only misfit
            misfit_check = swmisfit!(params, vel, shots, misfitobj; runparams=runparams)

            # Check that gradient is non zero
            @test !all(g -> g == 0.0, grad["vp"])
            # Check that misfits are non zero
            @test !(misfit ≈ 0.0)
            @test !(misfit_check ≈ 0.0)
            # Check that misfits are equivalent
            @test misfit ≈ misfit_check
        end

        @testset "Test 1D $(parall) swgradient! with compute misfit CCTSMisfit" begin
            # Physics
            c0 = 2000.0
            f0 = 10.0
            # Numerics
            nt = 1000
            nx = 101
            dx = 10.0
            dt = 0.99 * dx / c0
            halo = 20
            rcoef = 0.0001
            params, shots, misfitobj, vel = setup_constant_vel_1D_CPML(nt, dt, nx, dx, c0, f0, halo, rcoef; ccts=true)

            # Compute gradient and misfit
            gradparams = GradParameters(compute_misfit=true)
            grad, misfit = swgradient!(
                params,
                vel,
                shots,
                misfitobj;
                runparams=runparams,
                gradparams=gradparams
            )
            # Compute only misfit
            misfit_check = swmisfit!(params, vel, shots, misfitobj; runparams=runparams)

            # Check that gradient is non zero
            @test !all(g -> g == 0.0, grad["vp"])
            # Check that misfits are non zero
            @test !(misfit ≈ 0.0)
            @test !(misfit_check ≈ 0.0)
            # Check that misfits are equivalent
            @test misfit ≈ misfit_check
        end

        @testset "Test 1D $(parall) swgradient! with compute misfit and windowing" begin
            # Physics
            c0 = 2000.0
            f0 = 10.0
            # Numerics
            nt = 1000
            nx = 101
            dx = 10.0
            dt = 0.99 * dx / c0
            halo = 20
            rcoef = 0.0001
            params, shots, misfitobj, vel = setup_constant_vel_1D_CPML(nt, dt, nx, dx, c0, f0, halo, rcoef)
            push!(misfitobj[1].windows, Pair(500, 600))

            # Compute gradient and misfit
            gradparams = GradParameters(compute_misfit=true)
            grad, misfit = swgradient!(
                params,
                vel,
                shots,
                misfitobj;
                runparams=runparams,
                gradparams=gradparams
            )
            # Compute only misfit
            misfit_check = swmisfit!(params, vel, shots, misfitobj; runparams=runparams)

            # Check that gradient is non zero
            @test !all(g -> g == 0.0, grad["vp"])
            # Check that misfits are non zero
            @test !(misfit ≈ 0.0)
            @test !(misfit_check ≈ 0.0)
            # Check that misfits are equivalent
            @test misfit ≈ misfit_check
        end

        @testset "Test 2D $(parall) swgradient! with compute misfit" begin
            # Physics
            c0 = 2000.0
            f0 = 10.0
            # Numerics
            nt = 1000
            nx = ny = 101
            dx = dy = 10.0
            dt = 0.99 * dx / c0 / sqrt(2)
            halo = 20
            rcoef = 0.0001
            params, shots, misfitobj, vel = setup_constant_vel_2D_CPML(nt, dt, nx, ny, dx, dy, c0, f0, halo, rcoef)

            # Compute gradient and misfit
            gradparams = GradParameters(compute_misfit=true)
            grad, misfit = swgradient!(
                params,
                vel,
                shots,
                misfitobj;
                runparams=runparams,
                gradparams=gradparams
            )
            # Compute only misfit
            misfit_check = swmisfit!(params, vel, shots, misfitobj; runparams=runparams)

            # Check that gradient is non zero
            @test !all(g -> g == 0.0, grad["vp"])
            # Check that misfits are non zero
            @test !(misfit ≈ 0.0)
            @test !(misfit_check ≈ 0.0)
            # Check that misfits are equivalent
            @test misfit ≈ misfit_check
        end

        @testset "Test 3D $(parall) swgradient! with compute misfit" begin
            # Physics
            c0 = 2000.0
            f0 = 10.0
            # Numerics
            nt = 100
            nx = ny = nz = 81
            dx = dy = dz = 10.0
            dt = 0.99 * dx / c0 / sqrt(3)
            halo = 20
            rcoef = 0.0001
            params, shots, misfitobj, vel = setup_constant_vel_3D_CPML(nt, dt, nx, ny, nz, dx, dy, dz, c0, f0, halo, rcoef)

            # Compute gradient and misfit
            gradparams = GradParameters(compute_misfit=true)
            grad, misfit = swgradient!(
                params,
                vel,
                shots,
                misfitobj;
                runparams=runparams,
                gradparams=gradparams
            )
            # Compute only misfit
            misfit_check = swmisfit!(params, vel, shots, misfitobj; runparams=runparams)

            # Check that gradient is non zero
            @test !all(g -> g == 0.0, grad["vp"])
            # Check that misfits are non zero
            @test !(misfit ≈ 0.0)
            @test !(misfit_check ≈ 0.0)
            # Check that misfits are equivalent
            @test misfit ≈ misfit_check
        end

        @testset "Test 1D $(parall) swgradient! checkpointing" begin
            # Physics
            c0 = 2000.0
            f0 = 10.0
            # Numerics
            nt = 1000
            nx = 101
            dx = 10.0
            dt = 0.99 * dx / c0
            halo = 20
            rcoef = 0.0001
            params, shots, misfitobj, vel = setup_constant_vel_1D_CPML(nt, dt, nx, dx, c0, f0, halo, rcoef)

            # Solve gradient without checkpointing
            gradparams = GradParameters()
            grad = swgradient!(
                params,
                vel,
                shots,
                misfitobj;
                runparams=runparams,
                gradparams=gradparams
            )
            # Solve gradient with (optimal) checkpointing
            gradparams = GradParameters(check_freq=floor(Int, sqrt(nt)))
            grad_check = swgradient!(
                params,
                vel,
                shots,
                misfitobj;
                runparams=runparams,
                gradparams=gradparams
            )

            # Check that gradient is non zero
            @test !all(g -> g == 0.0, grad["vp"])
            # Check that computations are equivalent
            @test grad["vp"] ≈ grad_check["vp"]
        end

        @testset "Test 2D $(parall) swgradient! checkpointing" begin
            # Physics
            c0 = 2000.0
            f0 = 10.0
            # Numerics
            nt = 1000
            nx = ny = 101
            dx = dy = 10.0
            dt = 0.99 * dx / c0 / sqrt(2)
            halo = 20
            rcoef = 0.0001
            params, shots, misfitobj, vel = setup_constant_vel_2D_CPML(nt, dt, nx, ny, dx, dy, c0, f0, halo, rcoef)

            # Solve gradient without checkpointing
            gradparams = GradParameters()
            grad = swgradient!(
                params,
                vel,
                shots,
                misfitobj;
                runparams=runparams,
                gradparams=gradparams
            )
            # Solve gradient with (optimal) checkpointing
            gradparams = GradParameters(check_freq=floor(Int, sqrt(nt)))
            grad_check = swgradient!(
                params,
                vel,
                shots,
                misfitobj;
                runparams=runparams,
                gradparams=gradparams
            )

            # Check that gradient is non zero
            @test !all(g -> g == 0.0, grad["vp"])
            # Check that computations are equivalent
            @test grad["vp"] ≈ grad_check["vp"]
        end

        @testset "Test 3D $(parall) swgradient! checkpointing" begin
            # Physics
            c0 = 2000.0
            f0 = 10.0
            # Numerics
            nt = 150
            nx = ny = nz = 81
            dx = dy = dz = 10.0
            dt = 0.99 * dx / c0 / sqrt(3)
            halo = 20
            rcoef = 0.0001
            params, shots, misfitobj, vel = setup_constant_vel_3D_CPML(nt, dt, nx, ny, nz, dx, dy, dz, c0, f0, halo, rcoef)

            # Solve gradient without checkpointing
            gradparams = GradParameters()
            grad = swgradient!(
                params,
                vel,
                shots,
                misfitobj;
                runparams=runparams,
                gradparams=gradparams
            )
            # Solve gradient with (optimal) checkpointing
            gradparams = GradParameters(check_freq=floor(Int, sqrt(nt)))
            grad_check = swgradient!(
                params,
                vel,
                shots,
                misfitobj;
                runparams=runparams,
                gradparams=gradparams
            )

            # Check that gradient is non zero
            @test !all(g -> g == 0.0, grad["vp"])
            # Check that computations are equivalent
            @test grad["vp"] ≈ grad_check["vp"]
        end
    end

    end
end
