using Test
using DSP, NumericalIntegration, LinearAlgebra
using CUDA: CUDA
using Logging
using SeismicWaves

with_logger(ConsoleLogger(stderr, Logging.Info)) do
    test_backends = [:threads]
    # test GPU backend only if CUDA is functional
    if @isdefined(CUDA) && CUDA.functional()
        push!(test_backends, :CUDA)
    end
    if @isdefined(AMDGPU) && AMDGPU.functional()
        push!(test_backends, :AMDGPU)
    end

    for parall in test_backends
        @testset "Test 2D (P-SV) $(parall) swgradient! with compute misfit" begin
            # Physics
            ρ0 = 2100.0
            vp0 = 3210.5
            vs0 = vp0 / sqrt(3)
            λ0 = ρ0 * (vp0^2 - 2 * vs0^2)
            μ0 = ρ0 * vs0^2
            f0 = 12.0
            T = 0.075
            # Numerics
            nx = 380 # 211
            nz = 270 # 120
            dx = dz = 4.5 # meters
            dt = 0.0005
            nt = ceil(Int, T / dt)
            halo = 20
            rcoef = 0.0001
            params, shots, matprop = setup_constant_elastic_2D_CPML(nt, dt, nx, nz, dx, dz, ρ0, λ0, μ0, halo, rcoef, f0)

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
            @test !all(g -> g == 0.0, grad["rho"])
            @test !all(g -> g == 0.0, grad["lambda"])
            @test !all(g -> g == 0.0, grad["mu"])
            # Check that misfits are non zero
            @test !(misfit ≈ 0.0)
            @test !(misfit_check ≈ 0.0)
            # Check that misfits are equivalent
            @show misfit, misfit_check
            @test misfit ≈ misfit_check
        end
    end
end
