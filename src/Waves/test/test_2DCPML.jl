using Test
using DSP, NumericalIntegration, LinearAlgebra

using Waves
import Waves.Acoustic2D_Threads

using Logging, Plots, LinearAlgebra

@testset "Test analytical solution" begin
    error_logger = ConsoleLogger(stderr, Logging.Error)
    with_logger(error_logger) do
        @testset "Test constant unitary velocity halo 0" begin
            # constant velocity setup
            c0 = 1.0
            nt = 1500
            nx = ny = 501
            dx = dy = 0.02
            dt = sqrt(2) / (c0 * (1/dx + 1/dy)) / 2
            vel = c0 .* ones(nx, ny)

            # model
            model = IsotropicAcousticCPMLWaveModel2D(nt, dt, dx, dy, 0, 1.0, vel, false)
            # sources
            f0 = 1.0
            t0 = 4 / f0
            times = collect(range(0.0, step=dt, length=nt))
            possrcs = zeros(1, 2)
            possrcs[1, :] = [model.lx / 2, model.ly / 2]
            srctf = zeros(nt, 1)
            srctf[:, 1] .= rickersource1D.(times, t0, f0)
            # receivers
            posrecs = zeros(1, 2)
            posrecs[1, :] = [model.lx / 4, model.ly / 2]
            srcs = Sources(possrcs, srctf, f0)
            recs = Receivers(posrecs, nt)

            # numerical solution
            solve!(model, [srcs => recs], Waves.Acoustic2D_Threads)
            numerical_trace = recs.seismograms[:, 1]

            # Analytical solution
            times = collect(range(0.0, step=dt, length=nt+1))
            dist = norm(possrcs[1,:] .- posrecs[1,:])
            src = rickersource1D.(times, t0, f0)
            # Calculate Green's function
            G = times .* 0.
            for it = 1:nt+1
                # Heaviside function
                if (times[it] - dist / c0) >= 0
                    G[it] = 1. / (2π * c0^2 * sqrt((times[it]^2) - (dist^2 / (c0^2))))
                end
            end
            # Convolve with source term
            Gc = conv(G, src .* dt)
            Gc = Gc[2:nt+1]     # skip time 0

            @test length(numerical_trace) == length(Gc) == nt
            # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
            @test integrate(times[2:end], abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
        end

        @testset "Test constant unitary velocity halo 20" begin
            # constant velocity setup
            c0 = 1.0
            nt = 4500
            nx = ny = 501
            dx = dy = 0.02
            dt = sqrt(2) / (c0 * (1/dx + 1/dy)) / 2
            vel = c0 .* ones(nx, ny)
            halo = 20
            rcoef = 0.0001
            # model
            model = IsotropicAcousticCPMLWaveModel2D(nt, dt, dx, dy, halo, rcoef, vel, false)
            # sources
            f0 = 1.0
            t0 = 4 / f0
            times = collect(range(0.0, step=dt, length=nt))
            possrcs = zeros(1, 2)
            possrcs[1, :] = [model.lx / 2, model.ly / 2]
            srctf = zeros(nt, 1)
            srctf[:, 1] .= rickersource1D.(times, t0, f0)
            # receivers
            posrecs = zeros(1, 2)
            posrecs[1, :] = [model.lx / 4, model.ly / 2]
            srcs = Sources(possrcs, srctf, f0)
            recs = Receivers(posrecs, nt)

            # numerical solution
            solve!(model, [srcs => recs], Waves.Acoustic2D_Threads)
            numerical_trace = recs.seismograms[:, 1]

            # Analytical solution
            times = collect(range(0.0, step=dt, length=nt+1))
            dist = norm(possrcs[1,:] .- posrecs[1,:])
            src = rickersource1D.(times, t0, f0)
            # Calculate Green's function
            G = times .* 0.
            for it = 1:nt+1
                # Heaviside function
                if (times[it] - dist / c0) >= 0
                    G[it] = 1. / (2π * c0^2 * sqrt((times[it]^2) - (dist^2 / (c0^2))))
                end
            end
            # Convolve with source term
            Gc = conv(G, src .* dt)
            Gc = Gc[2:nt+1]     # skip time 0

            @test length(numerical_trace) == length(Gc) == nt
            # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
            @test integrate(times[2:end], abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
        end

        @testset "Test constant velocity (5 m/s) halo 0" begin
            # constant velocity setup
            c0 = 5.0
            nt = 1500
            nx = ny = 501
            dx = dy = 0.02
            dt = sqrt(2) / (c0 * (1/dx + 1/dy)) / 2
            vel = c0 .* ones(nx, ny)

            # model
            model = IsotropicAcousticCPMLWaveModel2D(nt, dt, dx, dy, 0, 1.0, vel, false)
            # sources
            f0 = 1.0
            t0 = 4 / f0
            times = collect(range(0.0, step=dt, length=nt))
            possrcs = zeros(1, 2)
            possrcs[1, :] = [model.lx / 2, model.ly / 2]
            srctf = zeros(nt, 1)
            srctf[:, 1] .= rickersource1D.(times, t0, f0)
            # receivers
            posrecs = zeros(1, 2)
            posrecs[1, :] = [model.lx / 4, model.ly / 2]
            srcs = Sources(possrcs, srctf, f0)
            recs = Receivers(posrecs, nt)

            # numerical solution
            solve!(model, [srcs => recs], Waves.Acoustic2D_Threads)
            numerical_trace = recs.seismograms[:, 1]

            # Analytical solution
            times = collect(range(0.0, step=dt, length=nt+1))
            dist = norm(possrcs[1,:] .- posrecs[1,:])
            src = rickersource1D.(times, t0, f0)
            # Calculate Green's function
            G = times .* 0.
            for it = 1:nt+1
                # Heaviside function
                if (times[it] - dist / c0) >= 0
                    G[it] = 1. / (2π * c0^2 * sqrt((times[it]^2) - (dist^2 / (c0^2))))
                end
            end
            # Convolve with source term
            Gc = conv(G, src .* dt)
            Gc = Gc[2:nt+1]     # skip time 0

            @test length(numerical_trace) == length(Gc) == nt
            # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
            @test integrate(times[2:end], abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
        end

        @testset "Test constant velocity (5 m/s) halo 20" begin
            # constant velocity setup
            c0 = 5.0
            nt = 4500
            nx = ny = 501
            dx = dy = 0.02
            dt = sqrt(2) / (c0 * (1/dx + 1/dy)) / 2
            vel = c0 .* ones(nx, ny)
            halo = 20
            rcoef = 0.0001
            # model
            model = IsotropicAcousticCPMLWaveModel2D(nt, dt, dx, dy, halo, rcoef, vel, false)
            # sources
            f0 = 1.0
            t0 = 4 / f0
            times = collect(range(0.0, step=dt, length=nt))
            possrcs = zeros(1, 2)
            possrcs[1, :] = [model.lx / 2, model.ly / 2]
            srctf = zeros(nt, 1)
            srctf[:, 1] .= rickersource1D.(times, t0, f0)
            # receivers
            posrecs = zeros(1, 2)
            posrecs[1, :] = [model.lx / 4, model.ly / 2]
            srcs = Sources(possrcs, srctf, f0)
            recs = Receivers(posrecs, nt)

            # numerical solution
            solve!(model, [srcs => recs], Waves.Acoustic2D_Threads)
            numerical_trace = recs.seismograms[:, 1]

            # Analytical solution
            times = collect(range(0.0, step=dt, length=nt+1))
            dist = norm(possrcs[1,:] .- posrecs[1,:])
            src = rickersource1D.(times, t0, f0)
            # Calculate Green's function
            G = times .* 0.
            for it = 1:nt+1
                # Heaviside function
                if (times[it] - dist / c0) >= 0
                    G[it] = 1. / (2π * c0^2 * sqrt((times[it]^2) - (dist^2 / (c0^2))))
                end
            end
            # Convolve with source term
            Gc = conv(G, src .* dt)
            Gc = Gc[2:nt+1]     # skip time 0

            @test length(numerical_trace) == length(Gc) == nt
            # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
            @test integrate(times[2:end], abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
        end
    end
end