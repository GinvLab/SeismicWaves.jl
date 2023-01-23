using Test
using DSP, NumericalIntegration, LinearAlgebra

using Waves
import Waves.Acoustic1D_Threads

using Logging

@testset "Test analytical solution" begin
    error_logger = ConsoleLogger(stderr, Logging.Error)
    with_logger(error_logger) do
        @testset "Test constant unitary velocity halo 0" begin
            # constant velocity setup
            c0 = 1.0
            nt = 2000
            nx = 1001
            dx = 0.01
            dt = dx / c0 / 2
            vel = c0 .* ones(nx)

            # model
            model = IsotropicAcousticCPMLWaveModel1D(nt, dt, dx, 0, 1.0, vel, 500)
            # sources
            f0 = 1.0
            t0 = 4 / f0
            times = collect(range(0.0, step=dt, length=nt))
            possrcs = zeros(1, 1)
            srctf = zeros(nt, 1)
            srctf[:, 1] .= rickersource1D.(times, t0, f0)
            possrcs[1, :] = [model.lx / 2]
            # receivers
            posrecs = zeros(1, 1)
            posrecs[1, :] = [model.lx / 4]
            srcs = Sources(possrcs, srctf, f0)
            recs = Receivers(posrecs, nt)

            # numerical solution
            solve!(model, [srcs => recs], Waves.Acoustic1D_Threads)
            numerical_trace = recs.seismograms[:, 1]

            # analytical solution
            times = collect(range(0.0, step=dt, length=nt + 1))
            dist = norm(possrcs[1, :] .- posrecs[1, :])
            src = rickersource1D.(times, t0, f0)
            # Calculate Green's function
            G = times .* 0.0
            for it = 1:nt+1
                # Heaviside function
                if (times[it] - dist / c0) >= 0
                    G[it] = 1.0 / (2 * c0)
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
            nt = 10000 * 5
            nx = 1001
            dx = 0.01
            dt = dx / c0 / 2
            vel = c0 .* ones(nx)

            halo = 20
            rcoef = 0.0001
            # model
            model = IsotropicAcousticCPMLWaveModel1D(nt, dt, dx, halo, rcoef, vel, 500)
            # sources
            f0 = 1.0
            t0 = 4 / f0
            times = collect(range(0.0, step=dt, length=nt))
            possrcs = zeros(1, 1)
            srctf = zeros(nt, 1)
            srctf[:, 1] .= rickersource1D.(times, t0, f0)
            possrcs[1, :] = [model.lx / 2]
            # receivers
            posrecs = zeros(1, 1)
            posrecs[1, :] = [model.lx / 4]
            srcs = Sources(possrcs, srctf, f0)
            recs = Receivers(posrecs, nt)

            # numerical solution
            solve!(model, [srcs => recs], Waves.Acoustic1D_Threads)
            numerical_trace = recs.seismograms[:, 1]

            # analytical solution
            times = collect(range(0.0, step=dt, length=nt + 1))
            dist = norm(possrcs[1, :] .- posrecs[1, :])
            src = rickersource1D.(times, t0, f0)
            # Calculate Green's function
            G = times .* 0.0
            for it = 1:nt+1
                # Heaviside function
                if (times[it] - dist / c0) >= 0
                    G[it] = 1.0 / (2 * c0)
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
            nt = 2000
            nx = 1001
            dx = 0.01
            dt = dx / c0 / 2
            vel = c0 .* ones(nx)

            # model
            model = IsotropicAcousticCPMLWaveModel1D(nt, dt, dx, 0, 1.0, vel, 500)
            # sources
            f0 = 1.0
            t0 = 4 / f0
            times = collect(range(0.0, step=dt, length=nt))
            possrcs = zeros(1, 1)
            srctf = zeros(nt, 1)
            srctf[:, 1] .= rickersource1D.(times, t0, f0)
            possrcs[1, :] = [model.lx / 2]
            # receivers
            posrecs = zeros(1, 1)
            posrecs[1, :] = [model.lx / 4]
            srcs = Sources(possrcs, srctf, f0)
            recs = Receivers(posrecs, nt)

            # numerical solution
            solve!(model, [srcs => recs], Waves.Acoustic1D_Threads)
            numerical_trace = recs.seismograms[:, 1]

            # analytical solution
            times = collect(range(0.0, step=dt, length=nt + 1))
            dist = norm(possrcs[1, :] .- posrecs[1, :])
            src = rickersource1D.(times, t0, f0)
            # Calculate Green's function
            G = times .* 0.0
            for it = 1:nt+1
                # Heaviside function
                if (times[it] - dist / c0) >= 0
                    G[it] = 1.0 / (2 * c0)
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
            nt = 10000 * 5
            nx = 1001
            dx = 0.01
            dt = dx / c0 / 2
            vel = c0 .* ones(nx)

            halo = 20
            rcoef = 0.0001
            # model
            model = IsotropicAcousticCPMLWaveModel1D(nt, dt, dx, halo, rcoef, vel, 500)
            # sources
            f0 = 1.0
            t0 = 4 / f0
            times = collect(range(0.0, step=dt, length=nt))
            possrcs = zeros(1, 1)
            srctf = zeros(nt, 1)
            srctf[:, 1] .= rickersource1D.(times, t0, f0)
            possrcs[1, :] = [model.lx / 2]
            # receivers
            posrecs = zeros(1, 1)
            posrecs[1, :] = [model.lx / 4]
            srcs = Sources(possrcs, srctf, f0)
            recs = Receivers(posrecs, nt)

            # numerical solution
            solve!(model, [srcs => recs], Waves.Acoustic1D_Threads)
            numerical_trace = recs.seismograms[:, 1]

            # analytical solution
            times = collect(range(0.0, step=dt, length=nt + 1))
            dist = norm(possrcs[1, :] .- posrecs[1, :])
            src = rickersource1D.(times, t0, f0)
            # Calculate Green's function
            G = times .* 0.0
            for it = 1:nt+1
                # Heaviside function
                if (times[it] - dist / c0) >= 0
                    G[it] = 1.0 / (2 * c0)
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