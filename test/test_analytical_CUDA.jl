using Test
using DSP, NumericalIntegration, LinearAlgebra

using SeismicWaves

include("utils.jl")

using Logging
error_logger = ConsoleLogger(stderr, Logging.Error)
with_logger(error_logger) do
    @testset "Test 1D analytical solution" begin
        @testset "Test constant unitary velocity halo 0" begin
            # constant velocity setup
            c0 = 1.0
            nt = 1000
            nx = 1001
            dx = 0.01
            dt = dx / c0
            halo = 0
            rcoef = 1.0
            f0 = 1.0
            params, srcs, recs, vel = setup_constant_vel_1D_CPML(nt, dt, nx, dx, c0, f0, halo, rcoef)
            times, Gc = analytical_solution_constant_vel_1D(c0, dt, nt, srcs, recs)

            # numerical solution
            forward!(params, vel, [srcs => recs]; use_GPU=true)
            numerical_trace = recs.seismograms[:, 1]

            @test length(numerical_trace) == length(Gc) == nt
            # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
            @test integrate(times, abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
        end

        @testset "Test constant unitary velocity halo 20" begin
            # constant velocity setup
            c0 = 1.0
            nt = 1000 * 2
            nx = 1001
            dx = 0.01
            dt = dx / c0
            halo = 20
            rcoef = 0.0001
            f0 = 1.0
            params, srcs, recs, vel = setup_constant_vel_1D_CPML(nt, dt, nx, dx, c0, f0, halo, rcoef)
            times, Gc = analytical_solution_constant_vel_1D(c0, dt, nt, srcs, recs)

            # numerical solution
            forward!(params, vel, [srcs => recs]; use_GPU=true)
            numerical_trace = recs.seismograms[:, 1]

            @test length(numerical_trace) == length(Gc) == nt
            # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
            @test integrate(times, abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
        end

        @testset "Test constant velocity (5 m/s) halo 0" begin
            # constant velocity setup
            c0 = 5.0
            nt = 1000
            nx = 1001
            dx = 0.01
            dt = dx / c0
            halo = 0
            rcoef = 1.0
            f0 = 5.0
            params, srcs, recs, vel = setup_constant_vel_1D_CPML(nt, dt, nx, dx, c0, f0, halo, rcoef)
            times, Gc = analytical_solution_constant_vel_1D(c0, dt, nt, srcs, recs)

            # numerical solution
            forward!(params, vel, [srcs => recs]; use_GPU=true)
            numerical_trace = recs.seismograms[:, 1]

            @test length(numerical_trace) == length(Gc) == nt
            # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
            @test integrate(times, abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
        end

        @testset "Test constant velocity (5 m/s) halo 20" begin
            # constant velocity setup
            c0 = 5.0
            nt = 1000 * 2
            nx = 1001
            dx = 0.01
            dt = dx / c0
            halo = 20
            rcoef = 0.0001
            f0 = 5.0
            params, srcs, recs, vel = setup_constant_vel_1D_CPML(nt, dt, nx, dx, c0, f0, halo, rcoef)
            times, Gc = analytical_solution_constant_vel_1D(c0, dt, nt, srcs, recs)
            # numerical solution
            forward!(params, vel, [srcs => recs]; use_GPU=true)
            numerical_trace = recs.seismograms[:, 1]

            @test length(numerical_trace) == length(Gc) == nt
            # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
            @test integrate(times, abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
        end
    end
    
    @testset "Test 2D analytical solution" begin
        @testset "Test constant unitary velocity halo 0" begin
            # constant velocity setup
            c0 = 1.0
            nt = 1200
            nx = ny = 401
            dx = dy = 0.02
            dt = sqrt(2) / (c0 * (1/dx + 1/dy)) / 2
            halo = 0
            rcoef = 1.0
            f0 = 1.0
            params, srcs, recs, vel = setup_constant_vel_2D_CPML(nt, dt, nx, ny, dx, dy, c0, f0, halo, rcoef)
            times, Gc = analytical_solution_constant_vel_2D(c0, dt, nt, srcs, recs)

            # numerical solution
            forward!(params, vel, [srcs => recs]; use_GPU=true)
            numerical_trace = recs.seismograms[:, 1]

            @test length(numerical_trace) == length(Gc) == nt
            # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
            @test integrate(times, abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
        end

        @testset "Test constant unitary velocity halo 20" begin
            # constant velocity setup
            c0 = 1.0
            nt = 1200 * 2
            nx = ny = 401
            dx = dy = 0.02
            dt = sqrt(2) / (c0 * (1/dx + 1/dy)) / 2
            halo = 20
            rcoef = 0.0001
            f0 = 1.0
            params, srcs, recs, vel = setup_constant_vel_2D_CPML(nt, dt, nx, ny, dx, dy, c0, f0, halo, rcoef)
            times, Gc = analytical_solution_constant_vel_2D(c0, dt, nt, srcs, recs)

            # numerical solution
            forward!(params, vel, [srcs => recs]; use_GPU=true)
            numerical_trace = recs.seismograms[:, 1]

            @test length(numerical_trace) == length(Gc) == nt
            # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
            @test integrate(times, abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
        end

        @testset "Test constant velocity (5 m/s) halo 0" begin
            # constant velocity setup
            c0 = 5.0
            nt = 1200
            nx = ny = 401
            dx = dy = 0.02
            dt = sqrt(2) / (c0 * (1/dx + 1/dy)) / 2
            halo = 0
            rcoef = 1.0
            f0 = 5.0
            params, srcs, recs, vel = setup_constant_vel_2D_CPML(nt, dt, nx, ny, dx, dy, c0, f0, halo, rcoef)
            times, Gc = analytical_solution_constant_vel_2D(c0, dt, nt, srcs, recs)

            # numerical solution
            forward!(params, vel, [srcs => recs]; use_GPU=true)
            numerical_trace = recs.seismograms[:, 1]

            @test length(numerical_trace) == length(Gc) == nt
            # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
            @test integrate(times, abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
        end

        @testset "Test constant velocity (5 m/s) halo 20" begin
            # constant velocity setup
            c0 = 5.0
            nt = 1200 * 2
            nx = ny = 401
            dx = dy = 0.02
            dt = sqrt(2) / (c0 * (1/dx + 1/dy)) / 2
            halo = 20
            rcoef = 0.0001
            f0 = 5.0
            params, srcs, recs, vel = setup_constant_vel_2D_CPML(nt, dt, nx, ny, dx, dy, c0, f0, halo, rcoef)
            times, Gc = analytical_solution_constant_vel_2D(c0, dt, nt, srcs, recs)

            # numerical solution
            forward!(params, vel, [srcs => recs]; use_GPU=true)
            numerical_trace = recs.seismograms[:, 1]

            @test length(numerical_trace) == length(Gc) == nt
            # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
            @test integrate(times, abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
        end
    end

    @testset "Test 3D analytical solution" begin
        @testset "Test constant unitary velocity halo 0" begin
            # constant velocity setup
            c0 = 1.0
            nt = 650
            nx = ny = nz = 101
            dx = dy = dz = 0.05
            dt = sqrt(3) / (c0 * (1/dx + 1/dy + 1/dz)) / 3
            halo = 0
            rcoef = 1.0
            f0 = 1.0
            params, srcs, recs, vel = setup_constant_vel_3D_CPML(nt, dt, nx, ny, nz, dx, dy, dz, c0, f0, halo, rcoef)
            times, Gc = analytical_solution_constant_vel_3D(c0, dt, nt, srcs, recs)

            # numerical solution
            forward!(params, vel, [srcs => recs]; use_GPU=true)
            numerical_trace = recs.seismograms[:, 1]

            @test length(numerical_trace) == length(Gc) == nt
            # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
            @test integrate(times, abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
        end

        @testset "Test constant unitary velocity halo 20" begin
            # constant velocity setup
            c0 = 1.0
            nt = 650 * 2
            nx = ny = nz = 101
            dx = dy = dz = 0.05
            dt = sqrt(3) / (c0 * (1/dx + 1/dy + 1/dz)) / 3
            halo = 20
            rcoef = 0.0001
            f0 = 1.0
            params, srcs, recs, vel = setup_constant_vel_3D_CPML(nt, dt, nx, ny, nz, dx, dy, dz, c0, f0, halo, rcoef)
            times, Gc = analytical_solution_constant_vel_3D(c0, dt, nt, srcs, recs)

            # numerical solution
            forward!(params, vel, [srcs => recs]; use_GPU=true)
            numerical_trace = recs.seismograms[:, 1]

            @test length(numerical_trace) == length(Gc) == nt
            # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
            @test integrate(times, abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
        end

        @testset "Test constant velocity (5 m/s) halo 0" begin
            # constant velocity setup
            c0 = 5.0
            nt = 650
            nx = ny = nz = 101
            dx = dy = dz = 0.05
            dt = sqrt(3) / (c0 * (1/dx + 1/dy + 1/dz)) / 3
            halo = 0
            rcoef = 1.0
            f0 = 5.0
            params, srcs, recs, vel = setup_constant_vel_3D_CPML(nt, dt, nx, ny, nz, dx, dy, dz, c0, f0, halo, rcoef)
            times, Gc = analytical_solution_constant_vel_3D(c0, dt, nt, srcs, recs)

            # numerical solution
            forward!(params, vel, [srcs => recs]; use_GPU=true)
            numerical_trace = recs.seismograms[:, 1]

            @test length(numerical_trace) == length(Gc) == nt
            # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
            @test integrate(times, abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
        end

        @testset "Test constant velocity (5 m/s) halo 20" begin
            # constant velocity setup
            c0 = 5.0
            nt = 650 * 2
            nx = ny = nz = 101
            dx = dy = dz = 0.05
            dt = sqrt(3) / (c0 * (1/dx + 1/dy + 1/dz)) / 3
            halo = 20
            rcoef = 0.0001
            f0 = 5.0
            params, srcs, recs, vel = setup_constant_vel_3D_CPML(nt, dt, nx, ny, nz, dx, dy, dz, c0, f0, halo, rcoef)
            times, Gc = analytical_solution_constant_vel_3D(c0, dt, nt, srcs, recs)

            # numerical solution
            forward!(params, vel, [srcs => recs]; use_GPU=true)
            numerical_trace = recs.seismograms[:, 1]

            @test length(numerical_trace) == length(Gc) == nt
            # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
            @test integrate(times, abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
        end
    end
end
