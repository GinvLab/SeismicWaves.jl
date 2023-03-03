using Test
using DSP, NumericalIntegration, LinearAlgebra

using SeismicWaves
import SeismicWaves.Acoustic1D_Threads
import SeismicWaves.Acoustic2D_Threads
import SeismicWaves.Acoustic3D_Threads

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
            vel = c0 .* ones(nx)

            halo = 0
            rcoef = 1.0
            # model
            model = IsotropicAcousticCPMLWaveModel1D(nt, dt, dx, halo, rcoef, vel)
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
            forward!(model, [srcs => recs], SeismicWaves.Acoustic1D_Threads)
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
            nt = 1000 * 2
            nx = 1001
            dx = 0.01
            dt = dx / c0
            vel = c0 .* ones(nx)

            halo = 20
            rcoef = 0.0001
            # model
            model = IsotropicAcousticCPMLWaveModel1D(nt, dt, dx, halo, rcoef, vel)
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
            forward!(model, [srcs => recs], SeismicWaves.Acoustic1D_Threads)
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
            nt = 1000
            nx = 1001
            dx = 0.01
            dt = dx / c0
            vel = c0 .* ones(nx)

            halo = 0
            rcoef = 1.0
            # model
            model = IsotropicAcousticCPMLWaveModel1D(nt, dt, dx, halo, rcoef, vel)
            # sources
            f0 = 5.0
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
            forward!(model, [srcs => recs], SeismicWaves.Acoustic1D_Threads)
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
            nt = 1000
            nx = 1001
            dx = 0.01
            dt = dx / c0
            vel = c0 .* ones(nx)

            halo = 20
            rcoef = 0.0001
            # model
            model = IsotropicAcousticCPMLWaveModel1D(nt, dt, dx, halo, rcoef, vel)
            # sources
            f0 = 5.0
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
            forward!(model, [srcs => recs], SeismicWaves.Acoustic1D_Threads)
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
    
    @testset "Test 2D analytical solution" begin
        @testset "Test constant unitary velocity halo 0" begin
            # constant velocity setup
            c0 = 1.0
            nt = 1200
            nx = ny = 401
            dx = dy = 0.02
            dt = sqrt(2) / (c0 * (1/dx + 1/dy)) / 2
            vel = c0 .* ones(nx, ny)

            halo = 0
            rcoef = 1.0
            # model
            model = IsotropicAcousticCPMLWaveModel2D(nt, dt, dx, dy, halo, rcoef, vel; freetop=false)
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
            forward!(model, [srcs => recs], SeismicWaves.Acoustic2D_Threads)
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
            nt = 1200 * 2
            nx = ny = 401
            dx = dy = 0.02
            dt = sqrt(2) / (c0 * (1/dx + 1/dy)) / 2
            vel = c0 .* ones(nx, ny)

            halo = 20
            rcoef = 0.0001
            # model
            model = IsotropicAcousticCPMLWaveModel2D(nt, dt, dx, dy, halo, rcoef, vel; freetop=false)
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
            forward!(model, [srcs => recs], SeismicWaves.Acoustic2D_Threads)
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
            nt = 1200
            nx = ny = 401
            dx = dy = 0.02
            dt = sqrt(2) / (c0 * (1/dx + 1/dy)) / 2
            vel = c0 .* ones(nx, ny)

            halo = 0
            rcoef = 1.0
            # model
            model = IsotropicAcousticCPMLWaveModel2D(nt, dt, dx, dy, halo, rcoef, vel; freetop=false)
            # sources
            f0 = 5.0
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
            forward!(model, [srcs => recs], SeismicWaves.Acoustic2D_Threads)
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
            nt = 1200 * 2
            nx = ny = 401
            dx = dy = 0.02
            dt = sqrt(2) / (c0 * (1/dx + 1/dy)) / 2
            vel = c0 .* ones(nx, ny)

            halo = 20
            rcoef = 0.0001
            # model
            model = IsotropicAcousticCPMLWaveModel2D(nt, dt, dx, dy, halo, rcoef, vel; freetop=false)
            # sources
            f0 = 5.0
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
            forward!(model, [srcs => recs], SeismicWaves.Acoustic2D_Threads)
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

    @testset "Test 3D analytical solution" begin
        @testset "Test constant unitary velocity halo 0" begin
            # constant velocity setup
            c0 = 1.0
            nt = 650
            nx = ny = nz = 101
            dx = dy = dz = 0.05
            dt = sqrt(3) / (c0 * (1/dx + 1/dy + 1/dz)) / 3
            vel = c0 .* ones(nx, ny, nz)
            halo = 0
            rcoef = 1.0
            # model
            model = IsotropicAcousticCPMLWaveModel3D(nt, dt, dx, dy, dz, halo, rcoef, vel; freetop=false)
            # sources
            f0 = 1.0
            t0 = 4 / f0
            times = collect(range(0.0, step=dt, length=nt))
            possrcs = zeros(1, 3)
            possrcs[1, :] = [model.lx / 2, model.ly / 2, model.lz / 2]
            srctf = zeros(nt, 1)
            srctf[:, 1] .= rickersource1D.(times, t0, f0)
            # receivers
            posrecs = zeros(1, 3)
            posrecs[1, :] = [model.lx / 4, model.ly / 2, model.lz / 2]
            srcs = Sources(possrcs, srctf, f0)
            recs = Receivers(posrecs, nt)

            # numerical solution
            forward!(model, [srcs => recs], SeismicWaves.Acoustic3D_Threads)
            numerical_trace = recs.seismograms[:, 1]

            # Analytical solution
            times = collect(range(0.0, step=dt, length=nt+1))
            dist = norm(possrcs[1,:] .- posrecs[1,:])
            src = rickersource1D.(times, t0, f0)
            # Calculate Green's function
            G = times .* 0.
            for it = 1:nt+1
                # Delta function
                if (times[it] - dist / c0) >= 0
                    G[it] = 1.0 / (4π * c0^2 * dist)
                    break
                end
            end
            # Convolve with source term
            Gc = conv(G, src)
            Gc = Gc[2:nt+1]     # skip time 0

            @test length(numerical_trace) == length(Gc) == nt
            # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
            @test integrate(times[2:end], abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
        end

        @testset "Test constant unitary velocity halo 20" begin
            # constant velocity setup
            c0 = 1.0
            nt = 650 * 2
            nx = ny = nz = 101
            dx = dy = dz = 0.05
            dt = sqrt(3) / (c0 * (1/dx + 1/dy + 1/dz)) / 3
            vel = c0 .* ones(nx, ny, nz)
            halo = 20
            rcoef = 0.0001
            # model
            model = IsotropicAcousticCPMLWaveModel3D(nt, dt, dx, dy, dz, halo, rcoef, vel; freetop=false)
            # sources
            f0 = 1.0
            t0 = 4 / f0
            times = collect(range(0.0, step=dt, length=nt))
            possrcs = zeros(1, 3)
            possrcs[1, :] = [model.lx / 2, model.ly / 2, model.lz / 2]
            srctf = zeros(nt, 1)
            srctf[:, 1] .= rickersource1D.(times, t0, f0)
            # receivers
            posrecs = zeros(1, 3)
            posrecs[1, :] = [model.lx / 4, model.ly / 2, model.lz / 2]
            srcs = Sources(possrcs, srctf, f0)
            recs = Receivers(posrecs, nt)

            # numerical solution
            forward!(model, [srcs => recs], SeismicWaves.Acoustic3D_Threads)
            numerical_trace = recs.seismograms[:, 1]

            # Analytical solution
            times = collect(range(0.0, step=dt, length=nt+1))
            dist = norm(possrcs[1,:] .- posrecs[1,:])
            src = rickersource1D.(times, t0, f0)
            # Calculate Green's function
            G = times .* 0.
            for it = 1:nt+1
                # Delta function
                if (times[it] - dist / c0) >= 0
                    G[it] = 1.0 / (4π * c0^2 * dist)
                    break
                end
            end
            # Convolve with source term
            Gc = conv(G, src)
            Gc = Gc[2:nt+1]     # skip time 0

            @test length(numerical_trace) == length(Gc) == nt
            # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
            @test integrate(times[2:end], abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
        end

        @testset "Test constant velocity (5 m/s) halo 0" begin
            # constant velocity setup
            c0 = 5.0
            nt = 650
            nx = ny = nz = 101
            dx = dy = dz = 0.05
            dt = sqrt(3) / (c0 * (1/dx + 1/dy + 1/dz)) / 3
            vel = c0 .* ones(nx, ny, nz)
            halo = 0
            rcoef = 1.0
            # model
            model = IsotropicAcousticCPMLWaveModel3D(nt, dt, dx, dy, dz, halo, rcoef, vel; freetop=false)
            # sources
            f0 = 5.0
            t0 = 4 / f0
            times = collect(range(0.0, step=dt, length=nt))
            possrcs = zeros(1, 3)
            possrcs[1, :] = [model.lx / 2, model.ly / 2, model.lz / 2]
            srctf = zeros(nt, 1)
            srctf[:, 1] .= rickersource1D.(times, t0, f0)
            # receivers
            posrecs = zeros(1, 3)
            posrecs[1, :] = [model.lx / 4, model.ly / 2, model.lz / 2]
            srcs = Sources(possrcs, srctf, f0)
            recs = Receivers(posrecs, nt)

            # numerical solution
            forward!(model, [srcs => recs], SeismicWaves.Acoustic3D_Threads)
            numerical_trace = recs.seismograms[:, 1]

            # Analytical solution
            times = collect(range(0.0, step=dt, length=nt+1))
            dist = norm(possrcs[1,:] .- posrecs[1,:])
            src = rickersource1D.(times, t0, f0)
            # Calculate Green's function
            G = times .* 0.
            for it = 1:nt+1
                # Delta function
                if (times[it] - dist / c0) >= 0
                    G[it] = 1.0 / (4π * c0^2 * dist)
                    break
                end
            end
            # Convolve with source term
            Gc = conv(G, src)
            Gc = Gc[2:nt+1]     # skip time 0

            @test length(numerical_trace) == length(Gc) == nt
            # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
            @test integrate(times[2:end], abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
        end

        @testset "Test constant velocity (5 m/s) halo 20" begin
            # constant velocity setup
            c0 = 5.0
            nt = 650 * 2
            nx = ny = nz = 101
            dx = dy = dz = 0.05
            dt = sqrt(3) / (c0 * (1/dx + 1/dy + 1/dz)) / 3
            vel = c0 .* ones(nx, ny, nz)
            halo = 20
            rcoef = 0.0001
            # model
            model = IsotropicAcousticCPMLWaveModel3D(nt, dt, dx, dy, dz, halo, rcoef, vel; freetop=false)
            # sources
            f0 = 5.0
            t0 = 4 / f0
            times = collect(range(0.0, step=dt, length=nt))
            possrcs = zeros(1, 3)
            possrcs[1, :] = [model.lx / 2, model.ly / 2, model.lz / 2]
            srctf = zeros(nt, 1)
            srctf[:, 1] .= rickersource1D.(times, t0, f0)
            # receivers
            posrecs = zeros(1, 3)
            posrecs[1, :] = [model.lx / 4, model.ly / 2, model.lz / 2]
            srcs = Sources(possrcs, srctf, f0)
            recs = Receivers(posrecs, nt)

            # numerical solution
            forward!(model, [srcs => recs], SeismicWaves.Acoustic3D_Threads)
            numerical_trace = recs.seismograms[:, 1]

            # Analytical solution
            times = collect(range(0.0, step=dt, length=nt+1))
            dist = norm(possrcs[1,:] .- posrecs[1,:])
            src = rickersource1D.(times, t0, f0)
            # Calculate Green's function
            G = times .* 0.
            for it = 1:nt+1
                # Delta function
                if (times[it] - dist / c0) >= 0
                    G[it] = 1.0 / (4π * c0^2 * dist)
                    break
                end
            end
            # Convolve with source term
            Gc = conv(G, src)
            Gc = Gc[2:nt+1]     # skip time 0

            @test length(numerical_trace) == length(Gc) == nt
            # test integral of absolute difference over time is less then a constant 1% error relative to the peak analytical solution
            @test integrate(times[2:end], abs.(numerical_trace .- Gc)) <= maximum(abs.(Gc)) * 0.01 * (dt * nt)
        end
    end
end
