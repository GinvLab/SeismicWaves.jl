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
        @testset "Test 1D $(parall) analytical solution" begin
            @testset "Test 1D $(parall) constant velocity constant density halo 0" begin
                # constant velocity setup
                c0 = 1000.0
                ρ0 = 1500.0
                nt = 500
                nx = 501
                dx = 2.5
                dt = dx / c0 * 6 / 7
                halo = 0
                rcoef = 1.0
                f0 = 5.0
                t0 = 2 / f0
                params, shots, matprop = setup_constant_vel_rho_1D_CPML(nt, dt, nx, dx, c0, ρ0, t0, f0, halo, rcoef)
                times, Gc = analytical_solution_constant_vel_constant_density_1D(c0, ρ0, dt, nt, t0, f0, shots[1].srcs, shots[1].recs)

                # numerical solution
                swforward!(params, matprop, shots; parall=parall)
                numerical_trace = shots[1].recs.seismograms[:, 1]

                @test length(numerical_trace) == length(Gc) == nt
                # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
                @test integrate(times, abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
            end

            @testset "Test 1D $(parall) constant velocity constant density halo 20" begin
                # constant velocity setup
                c0 = 1000.0
                ρ0 = 1500.0
                nt = 5000
                nx = 501
                dx = 2.5
                dt = dx / c0 * 6 / 7
                halo = 20
                rcoef = 0.0001
                f0 = 5.0
                t0 = 2 / f0
                params, shots, matprop = setup_constant_vel_rho_1D_CPML(nt, dt, nx, dx, c0, ρ0, t0, f0, halo, rcoef)
                times, Gc = analytical_solution_constant_vel_constant_density_1D(c0, ρ0, dt, nt, t0, f0, shots[1].srcs, shots[1].recs)

                # numerical solution
                swforward!(params, matprop, shots; parall=parall)
                numerical_trace = shots[1].recs.seismograms[:, 1]

                @test length(numerical_trace) == length(Gc) == nt
                # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
                @test integrate(times, abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
            end
        end

        @testset "Test 2D $(parall) analytical solution" begin
            @testset "Test 2D $(parall) constant velocity contant density halo 0" begin
                # constant velocity setup
                c0 = 1000.0
                ρ0 = 1500.0
                nt = 350
                nx = ny = 401
                dx = dy = 5.0
                dt = dx / c0 / sqrt(2) * 6 / 7
                halo = 0
                rcoef = 1.0
                f0 = 5.0
                t0 = 2 / f0
                params, shots, matprop = setup_constant_vel_rho_2D_CPML(nt, dt, nx, ny, dx, dy, c0, ρ0, t0, f0, halo, rcoef)
                times, Gc = analytical_solution_constant_vel_constant_density_2D(c0, ρ0, dt, nt, t0, f0, shots[1].srcs, shots[1].recs)

                # numerical solution
                swforward!(params, matprop, shots; parall=parall)
                numerical_trace = shots[1].recs.seismograms[:, 1]

                @test length(numerical_trace) == length(Gc) == nt
                # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
                @test integrate(times, abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
            end

            @testset "Test 2D $(parall) constant velocity halo 20" begin
                # constant velocity setup
                c0 = 1000.0
                ρ0 = 1500.0
                nt = 350 * 4
                nx = ny = 401
                dx = dy = 5.0
                dt = dx / c0 / sqrt(2) * 6 / 7
                halo = 20
                rcoef = 0.0001
                f0 = 5.0
                t0 = 2 / f0
                params, shots, matprop = setup_constant_vel_rho_2D_CPML(nt, dt, nx, ny, dx, dy, c0, ρ0, t0, f0, halo, rcoef)
                times, Gc = analytical_solution_constant_vel_constant_density_2D(c0, ρ0, dt, nt, t0, f0, shots[1].srcs, shots[1].recs)

                # numerical solution
                swforward!(params, matprop, shots; parall=parall)
                numerical_trace = shots[1].recs.seismograms[:, 1]

                @test length(numerical_trace) == length(Gc) == nt
                # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
                @test integrate(times, abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
            end
        end
    end
end
