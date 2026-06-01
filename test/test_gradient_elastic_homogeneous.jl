
with_logger(ConsoleLogger(stderr, Logging.Warn)) do
    test_backends = [:threads]
    # test GPU backend only if CUDA is functional
    if @isdefined(CUDA) && CUDA.functional()
        push!(test_backends, :CUDA)
    end
    if @isdefined(AMDGPU) && AMDGPU.functional()
        push!(test_backends, :AMDGPU)
    end


    @testset "Test gradient (elastic homogeneous)" begin

    for parall in test_backends
        # parallelisation
        runparams = RunParameters(parall=parall)

        @testset "Test 2D (P-SV) $(parall) swgradient! with compute misfit" begin
            # Physics
            ρ0 = 2100.0
            vp0 = 3210.5
            vs0 = vp0 / sqrt(3)
            λ0 = ρ0 * (vp0^2 - 2 * vs0^2)
            μ0 = ρ0 * vs0^2
            f0 = 12.0
            T = 0.2
            # Numerics
            nx = 380 # 211
            nz = 270 # 120
            dx = dz = 4.5 # meters
            dt = 0.0005
            nt = ceil(Int, T / dt)
            halo = 20
            rcoef = 0.0001
            params, shots, misfitobj, matprop = setup_constant_elastic_2D_CPML(nt, dt, nx, nz, dx, dz, ρ0, λ0, μ0, halo, rcoef, f0)

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
            misfit_check = swmisfit!(params, matprop, shots, misfitobj; runparams=runparams)

            # Check that gradient is non zero
            @test !all(g -> g == 0.0, grad["rho"])
            @test !all(g -> g == 0.0, grad["lambda"])
            @test !all(g -> g == 0.0, grad["mu"])
            # Check that misfits are non zero
            @test !(misfit ≈ 0.0)
            @test !(misfit_check ≈ 0.0)
            # Check that misfits are equivalent
            @test misfit ≈ misfit_check
        end

        @testset "Test 2D (P-SV) $(runparams.parall) swgradient! checkpointing" begin
            # Physics
            ρ0 = 2100.0
            vp0 = 3210.5
            vs0 = vp0 / sqrt(3)
            λ0 = ρ0 * (vp0^2 - 2 * vs0^2)
            μ0 = ρ0 * vs0^2
            f0 = 12.0
            T = 0.2
            # Numerics
            nx = 380 # 211
            nz = 270 # 120
            dx = dz = 4.5 # meters
            dt = 0.0005
            nt = ceil(Int, T / dt)
            halo = 20
            rcoef = 0.0001
            params, shots, misfitobj, matprop = setup_constant_elastic_2D_CPML(nt, dt, nx, nz, dx, dz, ρ0, λ0, μ0, halo, rcoef, f0)

            # Compute gradient and misfit with checkpointing
            gradparams = GradParameters(check_freq=floor(Int, sqrt(nt)),
                                        compute_misfit=true)
            grad, misfit = swgradient!(
                params,
                matprop,
                shots,
                misfitobj;
                runparams=runparams,
                gradparams=gradparams
            )
            # Compute gradient and misfit without checkpointing
            gradparams = GradParameters(compute_misfit=true)
            grad_check, misfit_check = swgradient!(
                params,
                matprop,
                shots,
                misfitobj;
                runparams=runparams,
                gradparams=gradparams
            )

            # Check that gradients are non zero
            @test !all(g -> g == 0.0, grad["rho"])
            @test !all(g -> g == 0.0, grad["lambda"])
            @test !all(g -> g == 0.0, grad["mu"])
            @test !all(g -> g == 0.0, grad_check["rho"])
            @test !all(g -> g == 0.0, grad_check["lambda"])
            @test !all(g -> g == 0.0, grad_check["mu"])
            # Check that misfits are non zero
            @test !(misfit ≈ 0.0)
            @test !(misfit_check ≈ 0.0)
            # Check that misfits are equivalent
            @test misfit ≈ misfit_check
            # Check that gradients are equivalent
            @test grad["rho"] ≈ grad_check["rho"]
            @test grad["lambda"] ≈ grad_check["lambda"]
            @test grad["mu"] ≈ grad_check["mu"]
        end

        @testset "Test 2D (P-SV) $(parall) swgradient! dot product test (reflective)" begin
            # Physics
            ρ0 = 2100.0
            vp0 = 3210.5
            vs0 = vp0 / sqrt(3)
            λ0 = ρ0 * (vp0^2 - 2 * vs0^2)
            μ0 = ρ0 * vs0^2
            f0 = 12.0
            T = 0.2
            # Numerics
            nx = 380 # 211
            nz = 270 # 120
            dx = dz = 4.5 # meters
            dt = 0.0005
            nt = ceil(Int, T / dt)
            halo = 0
            rcoef = 0.0001
            params, shots, misfitobj, matprop = setup_constant_elastic_2D_CPML(nt, dt, nx, nz, dx, dz, ρ0, λ0, μ0, halo, rcoef, f0)

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

            # Check that gradients are non zero
            @test !all(g -> g == 0.0, grad["rho"])
            @test !all(g -> g == 0.0, grad["lambda"])
            @test !all(g -> g == 0.0, grad["mu"])

            # Perturb lambda in random direction
            δm_rel = 1e-5
            Δm = rand(size(matprop.λ)...)
            Δm .= Δm ./ norm(Δm) .* (δm_rel * maximum(matprop.λ))
            matprop_perturbed = ElasticIsoMaterialProperties(; ρ=matprop.ρ, λ=matprop.λ .+ Δm, μ=matprop.μ)

            # Compute misfit for perturbed model
            misfit_perturbed = swmisfit!(params, matprop_perturbed, shots, misfitobj; runparams=runparams)
            # Compute dot product between gradient and perturbation
            rhs = dot(grad["lambda"], Δm)
            # Compute finite difference approximation of misfit change
            lhs = (misfit_perturbed - misfit)

            # Check that finite difference approximation is close to gradient dot perturbation
            @test isapprox(lhs, rhs; rtol=1e2*δm_rel)

            # Perturb mu in random direction
            δm_rel = 1e-5
            Δm = rand(size(matprop.μ)...)
            Δm .= Δm ./ norm(Δm) .* (δm_rel * maximum(matprop.μ))
            matprop_perturbed = ElasticIsoMaterialProperties(; ρ=matprop.ρ, λ=matprop.λ, μ=matprop.μ .+ Δm)
            # Compute misfit for perturbed model
            misfit_perturbed = swmisfit!(params, matprop_perturbed, shots, misfitobj; runparams=runparams)
            # Compute dot product between gradient and perturbation
            rhs = dot(grad["mu"], Δm)
            # Compute finite difference approximation of misfit change
            lhs = (misfit_perturbed - misfit)

            # Check that finite difference approximation is close to gradient dot perturbation
            @test isapprox(lhs, rhs; rtol=1e2*δm_rel)

            # Perturb density in random direction
            δm_rel = 1e-3
            Δm = rand(size(matprop.ρ)...)
            Δm .= Δm ./ norm(Δm) .* (δm_rel * maximum(matprop.ρ))
            matprop_perturbed = ElasticIsoMaterialProperties(; ρ=matprop.ρ .+ Δm, λ=matprop.λ, μ=matprop.μ)
            # Compute misfit for perturbed model
            misfit_perturbed = swmisfit!(params, matprop_perturbed, shots, misfitobj; runparams=runparams)
            # Compute dot product between gradient and perturbation
            rhs = dot(grad["rho"], Δm)
            # Compute finite difference approximation of misfit change
            lhs = (misfit_perturbed - misfit)
            
            # Check that finite difference approximation is close to gradient dot perturbation
            @test isapprox(lhs, rhs; rtol=1e2*δm_rel)
        end

        @testset "Test 2D (P-SV) $(parall) swgradient! dot product test (CPML)" begin
            # Physics
            ρ0 = 2100.0
            vp0 = 3210.5
            vs0 = vp0 / sqrt(3)
            λ0 = ρ0 * (vp0^2 - 2 * vs0^2)
            μ0 = ρ0 * vs0^2
            f0 = 12.0
            T = 0.2
            # Numerics
            nx = 380 # 211
            nz = 270 # 120
            dx = dz = 4.5 # meters
            dt = 0.0005
            nt = ceil(Int, T / dt)
            halo = 20
            rcoef = 0.0001
            params, shots, misfitobj, matprop = setup_constant_elastic_2D_CPML(nt, dt, nx, nz, dx, dz, ρ0, λ0, μ0, halo, rcoef, f0)

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

            # Check that gradients are non zero
            @test !all(g -> g == 0.0, grad["rho"])
            @test !all(g -> g == 0.0, grad["lambda"])
            @test !all(g -> g == 0.0, grad["mu"])

            # Perturb lambda in random direction
            δm_rel = 1e-5
            Δm = rand(size(matprop.λ)...)
            Δm .= Δm ./ norm(Δm) .* (δm_rel * maximum(matprop.λ))
            # Mute perturbation in CPML region
            Δm[1:halo+1, :] .= 0.0
            Δm[end-halo-1:end, :] .= 0.0
            Δm[:, 1:halo+1] .= 0.0
            Δm[:, end-halo-1:end] .= 0.0
            matprop_perturbed = ElasticIsoMaterialProperties(; ρ=matprop.ρ, λ=matprop.λ .+ Δm, μ=matprop.μ)

            # Compute misfit for perturbed model
            misfit_perturbed = swmisfit!(params, matprop_perturbed, shots, misfitobj; runparams=runparams)
            # Compute dot product between gradient and perturbation
            rhs = dot(grad["lambda"], Δm)
            # Compute finite difference approximation of misfit change
            lhs = (misfit_perturbed - misfit)

            # Check that finite difference approximation is close to gradient dot perturbation
            @test isapprox(lhs, rhs; rtol=1e2*δm_rel)

            # Perturb mu in random direction
            δm_rel = 1e-5
            Δm = rand(size(matprop.μ)...)
            Δm .= Δm ./ norm(Δm) .* (δm_rel * maximum(matprop.μ))
            # Mute perturbation in CPML region
            Δm[1:halo+1, :] .= 0.0
            Δm[end-halo-1:end, :] .= 0.0
            Δm[:, 1:halo+1] .= 0.0
            Δm[:, end-halo-1:end] .= 0.0
            matprop_perturbed = ElasticIsoMaterialProperties(; ρ=matprop.ρ, λ=matprop.λ, μ=matprop.μ .+ Δm)
            # Compute misfit for perturbed model
            misfit_perturbed = swmisfit!(params, matprop_perturbed, shots, misfitobj; runparams=runparams)
            # Compute dot product between gradient and perturbation
            rhs = dot(grad["mu"], Δm)
            # Compute finite difference approximation of misfit change
            lhs = (misfit_perturbed - misfit)

            # Check that finite difference approximation is close to gradient dot perturbation
            @test isapprox(lhs, rhs; rtol=1e2*δm_rel)

            # Perturb density in random direction
            δm_rel = 1e-3
            Δm = rand(size(matprop.ρ)...)
            Δm .= Δm ./ norm(Δm) .* (δm_rel * maximum(matprop.ρ))
            # Mute perturbation in CPML region
            Δm[1:halo+1, :] .= 0.0
            Δm[end-halo-1:end, :] .= 0.0
            Δm[:, 1:halo+1] .= 0.0
            Δm[:, end-halo-1:end] .= 0.0
            matprop_perturbed = ElasticIsoMaterialProperties(; ρ=matprop.ρ .+ Δm, λ=matprop.λ, μ=matprop.μ)
            # Compute misfit for perturbed model
            misfit_perturbed = swmisfit!(params, matprop_perturbed, shots, misfitobj; runparams=runparams)
            # Compute dot product between gradient and perturbation
            rhs = dot(grad["rho"], Δm)
            # Compute finite difference approximation of misfit change
            lhs = (misfit_perturbed - misfit)
            
            # Check that finite difference approximation is close to gradient dot perturbation
            @test isapprox(lhs, rhs; rtol=1e2*δm_rel)
        end

        @testset "Test 2D (P-SV) $(parall) swgradient! dot product test (freetop + CPML)" begin
            # Physics
            ρ0 = 2100.0
            vp0 = 3210.5
            vs0 = vp0 / sqrt(3)
            λ0 = ρ0 * (vp0^2 - 2 * vs0^2)
            μ0 = ρ0 * vs0^2
            f0 = 12.0
            T = 0.2
            # Numerics
            nx = 380 # 211
            nz = 270 # 120
            dx = dz = 4.5 # meters
            dt = 0.0005
            nt = ceil(Int, T / dt)
            
            halo = 20
            rcoef = 0.0001
            params, shots, misfitobj, matprop = setup_constant_elastic_2D_CPML(nt, dt, nx, nz, dx, dz, ρ0, λ0, μ0, halo, rcoef, f0; freetop=true)

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

            # Check that gradients are non zero
            @test !all(g -> g == 0.0, grad["rho"])
            @test !all(g -> g == 0.0, grad["lambda"])
            @test !all(g -> g == 0.0, grad["mu"])

            # Perturb lambda in random direction
            δm_rel = 1e-5
            Δm = rand(size(matprop.λ)...)
            Δm .= Δm ./ norm(Δm) .* (δm_rel * maximum(matprop.λ))
            # Mute perturbation in CPML region
            Δm[1:halo+1, :] .= 0.0
            Δm[end-halo-1:end, :] .= 0.0
            Δm[:, end-halo-1:end] .= 0.0
            matprop_perturbed = ElasticIsoMaterialProperties(; ρ=matprop.ρ, λ=matprop.λ .+ Δm, μ=matprop.μ)

            # Compute misfit for perturbed model
            misfit_perturbed = swmisfit!(params, matprop_perturbed, shots, misfitobj; runparams=runparams)
            # Compute dot product between gradient and perturbation
            rhs = dot(grad["lambda"], Δm)
            # Compute finite difference approximation of misfit change
            lhs = (misfit_perturbed - misfit)

            # Check that finite difference approximation is close to gradient dot perturbation
            @test isapprox(lhs, rhs; rtol=1e2*δm_rel)

            # Perturb mu in random direction
            δm_rel = 1e-5
            Δm = rand(size(matprop.μ)...)
            Δm .= Δm ./ norm(Δm) .* (δm_rel * maximum(matprop.μ))
            # Mute perturbation in CPML region
            Δm[1:halo+1, :] .= 0.0
            Δm[end-halo-1:end, :] .= 0.0
            Δm[:, end-halo-1:end] .= 0.0
            matprop_perturbed = ElasticIsoMaterialProperties(; ρ=matprop.ρ, λ=matprop.λ, μ=matprop.μ .+ Δm)
            # Compute misfit for perturbed model
            misfit_perturbed = swmisfit!(params, matprop_perturbed, shots, misfitobj; runparams=runparams)
            # Compute dot product between gradient and perturbation
            rhs = dot(grad["mu"], Δm)
            # Compute finite difference approximation of misfit change
            lhs = (misfit_perturbed - misfit)

            # Check that finite difference approximation is close to gradient dot perturbation
            @test isapprox(lhs, rhs; rtol=1e2*δm_rel)

            # Perturb density in random direction
            δm_rel = 1e-3
            Δm = rand(size(matprop.ρ)...)
            Δm .= Δm ./ norm(Δm) .* (δm_rel * maximum(matprop.ρ))
            # Mute perturbation in CPML region
            Δm[1:halo+1, :] .= 0.0
            Δm[end-halo-1:end, :] .= 0.0
            Δm[:, end-halo-1:end] .= 0.0
            matprop_perturbed = ElasticIsoMaterialProperties(; ρ=matprop.ρ .+ Δm, λ=matprop.λ, μ=matprop.μ)
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
