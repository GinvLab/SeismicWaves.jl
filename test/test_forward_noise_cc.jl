
with_logger(ConsoleLogger(stderr, Logging.Warn)) do
    test_backends = [:threads]
    # test GPU backend only if CUDA is functional
    if @isdefined(CUDA) && CUDA.functional()
        push!(test_backends, :CUDA)
    end
    if @isdefined(AMDGPU) && AMDGPU.functional()
        push!(test_backends, :AMDGPU)
    end

    @testset "Test forward noise CCs (elastic isotropic)" begin

    for parall in test_backends
        # parallelisation
        runparams = RunParameters(parall=parall)

        @testset "Test 2D (P-SV) $(parall) point noise source" begin
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
            params, shots, matprop = setup_constant_elastic_2D_noise_CPML(nt, dt, nx, nz, dx, dz, ρ0, λ0, μ0, halo, rcoef, f0)

            # Run forward
            swforward!(params, matprop, shots; runparams=runparams)
        end
    end

    end
end