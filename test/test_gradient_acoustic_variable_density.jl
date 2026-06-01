
with_logger(ConsoleLogger(stderr, Logging.Warn)) do
    test_backends = [:threads]
    # test GPU backend only if CUDA is functional
    if @isdefined(CUDA) && CUDA.functional()
        push!(test_backends, :CUDA)
    end
    if @isdefined(AMDGPU) && AMDGPU.functional()
        push!(test_backends, :AMDGPU)
    end

    @testset "Test gradient (acoustic VD)" begin

    for parall in test_backends
        runparams = RunParameters(parall=parall)

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
            dt = 0.99 * dx / c0 * 6 / 7
            halo = 20
            rcoef = 0.0001
            params, shots, misfitobj, matprop = setup_constant_vel_rho_1D_CPML(nt, dt, nx, dx, c0, ρ0, t0, f0, halo, rcoef)

            # Compute gradient and misfit
            gradparams = GradParameters(compute_misfit=true)
            grad, misfit = swgradient!(
                params,
                matprop,
                shots,
                misfitobj;
                runparams=runparams,
                gradparams=gradparams
            )
            # Compute only misfit
            misfit_check = swmisfit!(params, matprop, shots, misfitobj; runparams=runparams )

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
            dt = 0.99 * dx / c0 / sqrt(2) * 6 / 7
            halo = 20
            rcoef = 0.0001
            params, shots, misfitobj, matprop = setup_constant_vel_rho_2D_CPML(nt, dt, nx, ny, dx, dy, c0, ρ0, t0, f0, halo, rcoef)

            # Compute gradient and misfit
            gradparams = GradParameters(compute_misfit=true)
            grad, misfit = swgradient!(
                params,
                matprop,
                shots,
                misfitobj;
                runparams=runparams,
                gradparams=gradparams
            )
            # Compute only misfit
            misfit_check = swmisfit!(params, matprop, shots, misfitobj; runparams=runparams )

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
            dt = 0.99 * dx / c0 * 6 / 7
            halo = 20
            rcoef = 0.0001
            params, shots, misfitobj, matprop = setup_constant_vel_rho_1D_CPML(nt, dt, nx, dx, c0, ρ0, t0, f0, halo, rcoef)

            # Solve gradient without checkpointing
            gradparams = GradParameters()
            grad = swgradient!(
                params,
                matprop,
                shots,
                misfitobj;
                runparams=runparams,
                gradparams=gradparams
            )
            # Solve gradient with (optimal) checkpointing
            gradparams = GradParameters(check_freq=floor(Int, sqrt(nt)))
            grad_check = swgradient!(
                params,
                matprop,
                shots,
                misfitobj;
                runparams=runparams,
                gradparams=gradparams
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
            dt = 0.99 * dx / c0 / sqrt(2) * 6 / 7
            halo = 20
            rcoef = 0.0001
            params, shots, misfitobj, matprop = setup_constant_vel_rho_2D_CPML(nt, dt, nx, ny, dx, dy, c0, ρ0, t0, f0, halo, rcoef)

            # Solve gradient without checkpointing
            gradparams = GradParameters()
            grad = swgradient!(
                params,
                matprop,
                shots,
                misfitobj;
                runparams=runparams,
                gradparams=gradparams
            )
            # Solve gradient with (optimal) checkpointing
            gradparams = GradParameters(check_freq=floor(Int, sqrt(nt)))
            grad_check = swgradient!(
                params,
                matprop,
                shots,
                misfitobj;
                runparams=runparams,
                gradparams=gradparams
            )

            # Check that gradient is non zero
            @test !all(g -> g == 0.0, grad["vp"])
            @test !all(g -> g == 0.0, grad["rho"])
            # Check that computations are equivalent
            @test grad["vp"] ≈ grad_check["vp"]
            @test grad["rho"] ≈ grad_check["rho"]
        end

        @testset "Test 1D $(parall) swgradient! dot product test (reflective)" begin
            # Physics
            c0 = 2000.0
            ρ0 = 1500.0
            f0 = 10.0
            t0 = 2 / f0
            # Numerics
            nt = 1100
            nx = 101
            dx = 10.0
            dt = 0.9 * dx / c0 * 6 / 7
            halo = 0
            rcoef = 0.0001
            params, shots, misfitobj, matprop = setup_constant_vel_rho_1D_CPML(nt, dt, nx, dx, c0, ρ0, t0, f0, halo, rcoef)

            # Solve gradient without checkpointing
            gradparams = GradParameters(compute_misfit=true)
            grad, misfit = swgradient!(
                params,
                matprop,
                shots,
                misfitobj;
                runparams=runparams,
                gradparams=gradparams
            )

            # Check that gradient is non zero
            @test !all(g -> g == 0.0, grad["vp"])
            @test !all(g -> g == 0.0, grad["rho"])

            # Perturb velocity model in random direction
            δm_rel = 1e-5
            Δm = rand(size(matprop.vp)...)
            Δm .= Δm ./ norm(Δm) .* (δm_rel * maximum(matprop.vp))
            matprop_perturbed = VpRhoAcousticVDMaterialProperties(matprop.vp .+ Δm, matprop.rho)

            # Compute misfit for perturbed model
            misfit_perturbed = swmisfit!(params, matprop_perturbed, shots, misfitobj; runparams=runparams)
            # Compute dot product between gradient and perturbation
            rhs = dot(grad["vp"], Δm)
            # Compute finite difference approximation of misfit change
            lhs = (misfit_perturbed - misfit)

            # Check that finite difference approximation is close to gradient dot perturbation
            @test isapprox(lhs, rhs; rtol=1e2*δm_rel)

            # Perturb density model in random direction
            δm_rel = 1e-3
            Δm = rand(size(matprop.rho)...)
            Δm .= Δm ./ norm(Δm) .* (δm_rel * maximum(matprop.rho))
            matprop_perturbed = VpRhoAcousticVDMaterialProperties(matprop.vp, matprop.rho .+ Δm)

            # Compute misfit for perturbed model
            misfit_perturbed = swmisfit!(params, matprop_perturbed, shots, misfitobj; runparams=runparams)
            # Compute dot product between gradient and perturbation
            rhs = dot(grad["rho"], Δm)
            # Compute finite difference approximation of misfit change
            lhs = (misfit_perturbed - misfit)

            # Check that finite difference approximation is close to gradient dot perturbation
            @test isapprox(lhs, rhs; rtol=1e2*δm_rel)
        end

        @testset "Test 1D $(parall) swgradient! dot product test (CPML)" begin
            # Physics
            c0 = 2000.0
            ρ0 = 1500.0
            f0 = 10.0
            t0 = 2 / f0
            # Numerics
            nt = 1100
            nx = 101
            dx = 10.0
            dt = 0.9 * dx / c0 * 6 / 7
            halo = 20
            rcoef = 0.0001
            params, shots, misfitobj, matprop = setup_constant_vel_rho_1D_CPML(nt, dt, nx, dx, c0, ρ0, t0, f0, halo, rcoef)

            # Solve gradient without checkpointing
            gradparams = GradParameters(compute_misfit=true)
            grad, misfit = swgradient!(
                params,
                matprop,
                shots,
                misfitobj;
                runparams=runparams,
                gradparams=gradparams
            )

            # Check that gradient is non zero
            @test !all(g -> g == 0.0, grad["vp"])
            @test !all(g -> g == 0.0, grad["rho"])

            # Perturb velocity model in random direction
            δm_rel = 1e-5
            Δm = rand(size(matprop.vp)...)
            # Exclude CPML region from perturbation
            Δm[1:halo+1] .= 0.0
            Δm[end-halo-1:end] .= 0.0
            Δm .= Δm ./ norm(Δm) .* (δm_rel * maximum(matprop.vp))
            matprop_perturbed = VpRhoAcousticVDMaterialProperties(matprop.vp .+ Δm, matprop.rho)

            # Compute misfit for perturbed model
            misfit_perturbed = swmisfit!(params, matprop_perturbed, shots, misfitobj; runparams=runparams)
            # Compute dot product between gradient and perturbation
            rhs = dot(grad["vp"], Δm)
            # Compute finite difference approximation of misfit change
            lhs = (misfit_perturbed - misfit)

            # Check that finite difference approximation is close to gradient dot perturbation
            @test isapprox(lhs, rhs; rtol=1e2*δm_rel)

            # Perturb density model in random direction
            δm_rel = 1e-3
            Δm = rand(size(matprop.rho)...)
            # Exclude CPML region from perturbation
            Δm[1:halo+1] .= 0.0
            Δm[end-halo-1:end] .= 0.0
            Δm .= Δm ./ norm(Δm) .* (δm_rel * maximum(matprop.rho))
            matprop_perturbed = VpRhoAcousticVDMaterialProperties(matprop.vp, matprop.rho .+ Δm)

            # Compute misfit for perturbed model
            misfit_perturbed = swmisfit!(params, matprop_perturbed, shots, misfitobj; runparams=runparams)
            # Compute dot product between gradient and perturbation
            rhs = dot(grad["rho"], Δm)
            # Compute finite difference approximation of misfit change
            lhs = (misfit_perturbed - misfit)

            # Check that finite difference approximation is close to gradient dot perturbation
            @test isapprox(lhs, rhs; rtol=1e2*δm_rel)
        end

        @testset "Test 2D $(parall) swgradient! dot product test (reflective)" begin
            # Physics
            c0 = 2000.0
            ρ0 = 1500.0
            f0 = 10.0
            t0 = 2 / f0
            # Numerics
            nt = 1100
            nx = ny = 101
            dx = dy = 10.0
            dt = 0.9 * dx / c0 / sqrt(2) * 6 / 7
            halo = 0
            rcoef = 0.0001
            params, shots, misfitobj, matprop = setup_constant_vel_rho_2D_CPML(nt, dt, nx, ny, dx, dy, c0, ρ0, t0, f0, halo, rcoef)

            # Solve gradient without checkpointing
            gradparams = GradParameters(compute_misfit=true)
            grad, misfit = swgradient!(
                params,
                matprop,
                shots,
                misfitobj;
                runparams=runparams,
                gradparams=gradparams
            )

            # Check that gradient is non zero
            @test !all(g -> g == 0.0, grad["vp"])
            @test !all(g -> g == 0.0, grad["rho"])

            # Perturb velocity model in random direction
            δm_rel = 1e-5
            Δm = rand(size(matprop.vp)...)
            Δm .= Δm ./ norm(Δm) .* (δm_rel * maximum(matprop.vp))
            matprop_perturbed = VpRhoAcousticVDMaterialProperties(matprop.vp .+ Δm, matprop.rho)

            # Compute misfit for perturbed model
            misfit_perturbed = swmisfit!(params, matprop_perturbed, shots, misfitobj; runparams=runparams)
            # Compute dot product between gradient and perturbation
            rhs = dot(grad["vp"], Δm)
            # Compute finite difference approximation of misfit change
            lhs = (misfit_perturbed - misfit)

            # Check that finite difference approximation is close to gradient dot perturbation
            @test isapprox(lhs, rhs; rtol=1e2*δm_rel)

            # Perturb density model in random direction
            δm_rel = 1e-3
            Δm = rand(size(matprop.rho)...)
            Δm .= Δm ./ norm(Δm) .* (δm_rel * maximum(matprop.rho))
            matprop_perturbed = VpRhoAcousticVDMaterialProperties(matprop.vp, matprop.rho .+ Δm)

            # Compute misfit for perturbed model
            misfit_perturbed = swmisfit!(params, matprop_perturbed, shots, misfitobj; runparams=runparams)
            # Compute dot product between gradient and perturbation
            rhs = dot(grad["rho"], Δm)
            # Compute finite difference approximation of misfit change
            lhs = (misfit_perturbed - misfit)

            # Check that finite difference approximation is close to gradient dot perturbation
            @test isapprox(lhs, rhs; rtol=1e2*δm_rel)
        end

        @testset "Test 2D $(parall) swgradient! dot product test (CPML)" begin
            # Physics
            c0 = 2000.0
            ρ0 = 1500.0
            f0 = 10.0
            t0 = 2 / f0
            # Numerics
            nt = 1100
            nx = ny = 101
            dx = dy = 10.0
            dt = 0.9 * dx / c0 / sqrt(2) * 6 / 7
            halo = 20
            rcoef = 0.0001
            params, shots, misfitobj, matprop = setup_constant_vel_rho_2D_CPML(nt, dt, nx, ny, dx, dy, c0, ρ0, t0, f0, halo, rcoef)

            # Solve gradient without checkpointing
            gradparams = GradParameters(compute_misfit=true)
            grad, misfit = swgradient!(
                params,
                matprop,
                shots,
                misfitobj;
                runparams=runparams,
                gradparams=gradparams
            )

            # Check that gradient is non zero
            @test !all(g -> g == 0.0, grad["vp"])
            @test !all(g -> g == 0.0, grad["rho"])

            # Perturb velocity model in random direction
            δm_rel = 1e-5
            Δm = rand(size(matprop.vp)...)
            # Exclude CPML region from perturbation
            Δm[1:halo+1, :] .= 0.0
            Δm[end-halo-1:end, :] .= 0.0
            Δm[:, 1:halo+1] .= 0.0
            Δm[:, end-halo-1:end] .= 0.0
            Δm .= Δm ./ norm(Δm) .* (δm_rel * maximum(matprop.vp))
            matprop_perturbed = VpRhoAcousticVDMaterialProperties(matprop.vp .+ Δm, matprop.rho)

            # Compute misfit for perturbed model
            misfit_perturbed = swmisfit!(params, matprop_perturbed, shots, misfitobj; runparams=runparams)
            # Compute dot product between gradient and perturbation
            rhs = dot(grad["vp"], Δm)
            # Compute finite difference approximation of misfit change
            lhs = (misfit_perturbed - misfit)

            # Check that finite difference approximation is close to gradient dot perturbation
            @test isapprox(lhs, rhs; rtol=1e2*δm_rel)

            # Perturb density model in random direction
            δm_rel = 1e-3
            Δm = rand(size(matprop.rho)...)
            # Exclude CPML region from perturbation
            Δm[1:halo+1, :] .= 0.0
            Δm[end-halo-1:end, :] .= 0.0
            Δm[:, 1:halo+1] .= 0.0
            Δm[:, end-halo-1:end] .= 0.0
            Δm .= Δm ./ norm(Δm) .* (δm_rel * maximum(matprop.rho))
            matprop_perturbed = VpRhoAcousticVDMaterialProperties(matprop.vp, matprop.rho .+ Δm)

            # Compute misfit for perturbed model
            misfit_perturbed = swmisfit!(params, matprop_perturbed, shots, misfitobj; runparams=runparams)
            # Compute dot product between gradient and perturbation
            rhs = dot(grad["rho"], Δm)
            # Compute finite difference approximation of misfit change
            lhs = (misfit_perturbed - misfit)

            # Check that finite difference approximation is close to gradient dot perturbation
            @test isapprox(lhs, rhs; rtol=1e2*δm_rel)
        end
    end

    end
end
