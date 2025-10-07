
with_logger(ConsoleLogger(stderr, Logging.Warn)) do
    test_backends = [:serial, :threads]
    # test GPU backend only if CUDA is functional
    if @isdefined(CUDA) && CUDA.functional()
        push!(test_backends, :CUDA)
    end
    if @isdefined(AMDGPU) && AMDGPU.functional()
        push!(test_backends, :AMDGPU)
    end

    @testset "Test analytical vs numerical solution (acoustic CD)" begin

    for parall in test_backends
        # parallelisation
        runparams = RunParameters(parall=parall)

        @testset "Test 1D $(parall) analytical solution" begin
            @testset "Test 1D $(parall) constant velocity halo 0" begin
                # constant velocity setup
                c0 = 1000.0
                nt = 500
                nx = 501
                dx = 2.5
                dt = 0.99 * dx / c0
                halo = 0
                rcoef = 1.0
                f0 = 5.0
                params, shots, _, vel = setup_constant_vel_1D_CPML(nt, dt, nx, dx, c0, f0, halo, rcoef)
                times, Gc = analytical_solution_constant_vel_1D(c0, dt, nt, shots[1].srcs, shots[1].recs)
              
                # numerical solution
                swforward!(params, vel, shots; runparams=runparams)
                numerical_trace = shots[1].recs.seismograms[:, 1]

                @test length(numerical_trace) == length(Gc) == nt
                # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
                @test integrate(times, abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
            end

            @testset "Test 1D $(parall) constant velocity halo 20" begin
                # constant velocity setup
                c0 = 1000.0
                nt = 1000
                nx = 501
                dx = 2.5
                dt = 0.99 * dx / c0
                halo = 20
                rcoef = 0.0001
                f0 = 5.0
                params, shots, _, vel = setup_constant_vel_1D_CPML(nt, dt, nx, dx, c0, f0, halo, rcoef)
                times, Gc = analytical_solution_constant_vel_1D(c0, dt, nt, shots[1].srcs, shots[1].recs)
                # numerical solution
                swforward!(params, vel, shots; runparams=runparams)
                numerical_trace = shots[1].recs.seismograms[:, 1]

                @test length(numerical_trace) == length(Gc) == nt
                # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
                @test integrate(times, abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
            end
        end

        @testset "Test 2D $(parall) analytical solution" begin
            @testset "Test 2D $(parall) constant velocity halo 0" begin
                # constant velocity setup
                c0 = 1000.0
                nt = 700
                nx = ny = 801
                dx = dy = 2.5
                dt = dx / c0 / sqrt(2)
                halo = 0
                rcoef = 1.0
                f0 = 5.0
                params, shots, _, vel = setup_constant_vel_2D_CPML(nt, dt, nx, ny, dx, dy, c0, f0, halo, rcoef)
                times, Gc = analytical_solution_constant_vel_2D(c0, dt, nt, shots[1].srcs, shots[1].recs)

                # numerical solution
                swforward!(params, vel, shots; runparams=runparams)
                numerical_trace = shots[1].recs.seismograms[:, 1]

                @test length(numerical_trace) == length(Gc) == nt
                # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
                @test integrate(times, abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
            end

            @testset "Test 2D $(parall) constant velocity halo 20" begin
                # constant velocity setup
                c0 = 1000.0
                nt = 1400
                nx = ny = 801
                dx = dy = 2.5
                dt = 0.99 * dx / c0 / sqrt(2)
                halo = 20
                rcoef = 0.0001
                f0 = 5.0
                params, shots, _, vel = setup_constant_vel_2D_CPML(nt, dt, nx, ny, dx, dy, c0, f0, halo, rcoef)
                times, Gc = analytical_solution_constant_vel_2D(c0, dt, nt, shots[1].srcs, shots[1].recs)

                # numerical solution
                swforward!(params, vel, shots; runparams=runparams)
                numerical_trace = shots[1].recs.seismograms[:, 1]

                @test length(numerical_trace) == length(Gc) == nt
                # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
                @test integrate(times, abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
            end
        end

        @testset "Test 3D $(parall) analytical solution" begin
            @testset "Test 3D $(parall) constant velocity halo 0" begin
                # constant velocity setup
                c0 = 1000.0
                nt = 200
                nx = ny = nz = 121
                dx = dy = dz = 8.0
                dt = 0.99 * dx / c0 / sqrt(3)
                halo = 0
                rcoef = 1.0
                f0 = 5.0
                params, shots, _, vel = setup_constant_vel_3D_CPML(nt, dt, nx, ny, nz, dx, dy, dz, c0, f0, halo, rcoef)
                times, Gc = analytical_solution_constant_vel_3D(c0, dt, nt, shots[1].srcs, shots[1].recs)

                # numerical solution
                swforward!(params, vel, shots; runparams=runparams)
                numerical_trace = shots[1].recs.seismograms[:, 1]

                @test length(numerical_trace) == length(Gc) == nt
                # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
                @test integrate(times, abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
            end

            @testset "Test 3D $(parall) constant velocity halo 20" begin
                # constant velocity setup
                c0 = 1000.0
                nt = 400
                nx = ny = nz = 121
                dx = dy = dz = 8.0
                dt = 0.99 * dx / c0 / sqrt(3)
                halo = 20
                rcoef = 0.0001
                f0 = 5.0
                params, shots, _, vel = setup_constant_vel_3D_CPML(nt, dt, nx, ny, nz, dx, dy, dz, c0, f0, halo, rcoef)
                times, Gc = analytical_solution_constant_vel_3D(c0, dt, nt, shots[1].srcs, shots[1].recs)

                # numerical solution
                swforward!(params, vel, shots; runparams=runparams)
                numerical_trace = shots[1].recs.seismograms[:, 1]

                @test length(numerical_trace) == length(Gc) == nt
                # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
                @test integrate(times, abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
            end
        end
    end

    end
end
