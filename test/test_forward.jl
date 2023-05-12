using Test
using DSP, NumericalIntegration, LinearAlgebra
using CUDA: CUDA

using SeismicWaves

include("utils/setup_models.jl")

using Logging
error_logger = ConsoleLogger(stderr, Logging.Error)
with_logger(error_logger) do
    test_backends = [:serial, :threads]
    # test GPU backend only if CUDA is functional
    if CUDA.functional()
        push!(test_backends, :GPU)
    end

    for parall in test_backends
        @testset "Test 1D $(parall) single-source multiple-receivers CPML" begin
            #  velocity setup
            c0 = 1000.0
            c0max = 1300.0
            nx = 501
            r = 100
            vel = gaussian_vel_1D(nx, c0, c0max, r)
            matprop = VpAcousticCDMaterialProperty(vel)
            # numerics
            nt = 500
            dx = 2.5
            lx = (nx-1) * dx
            dt = dx / c0max
            halo = 20
            rcoef = 0.0001
            f0 = 5.0
            # wave simulation
            params = InputParametersAcoustic(nt, dt, [nx], [dx],
                     CPMLBoundaryConditionParameters(halo, rcoef, false))
            wavesim = build_wavesim(params; parall=parall)

            # single source at 10 grid points from CPML boundary
            times = collect(range(0.0; step=dt, length=nt))
            xsrc = (halo + 10) * dx
            srcs = ScalarSources(
                reshape([xsrc], 1, 1),
                reshape(rickersource1D.(times, 2/f0, f0), nt, 1),
                f0
            )
            # multiple receivers at different distances
            nrecs = 4
            dist = 50
            recs = ScalarReceivers(
                reshape(xsrc .+ dist .* collect(1:nrecs) .* fill(dx, nrecs), nrecs, 1),
                nt
            )

            # compute forward
            swforward!(wavesim, matprop, [Shot(srcs, recs)])
            # save seismograms
            res = copy(recs.seismograms)

            # same receivers, but positions reversed
            reverse!(recs.positions, dims=1)
            # compute forward
            swforward!(wavesim, matprop, [Shot(srcs, recs)])
            
            # test that seismograms are the same
            for i in 1:nrecs
                @test res[:, i] ≈ recs.seismograms[:, nrecs-i+1]
            end
        end

        @testset "Test 1D $(parall) multiple-sources multiple-receivers CPML" begin
            #  velocity setup
            c0 = 1000.0
            c0max = 1300.0
            nx = 501
            r = 100
            vel = gaussian_vel_1D(nx, c0, c0max, r)
            matprop = VpAcousticCDMaterialProperty(vel)
            # numerics
            nt = 500
            dx = 2.5
            lx = (nx-1) * dx
            dt = dx / c0max
            halo = 20
            rcoef = 0.0001
            f0 = 5.0
            # wave simulation
            params = InputParametersAcoustic(nt, dt, [nx], [dx],
                     CPMLBoundaryConditionParameters(halo, rcoef, false))
            wavesim = build_wavesim(params; parall=parall)

            # multiple sources at differece distances
            times = collect(range(0.0; step=dt, length=nt))
            xsrc = (halo + 10) * dx
            nsrcs = 3
            distsrcs = 10
            srcs = ScalarSources(
                reshape(xsrc .+ distsrcs .* collect(1:nsrcs) .* fill(dx, nsrcs), nsrcs, 1),
                repeat(rickersource1D.(times, 2/f0, f0), 1, nsrcs),
                f0
            )
            # multiple receivers at different distances
            nrecs = 6
            distrecs = 50
            recs = ScalarReceivers(
                reshape(xsrc .+ distrecs .* collect(1:nrecs) .* fill(dx, nrecs), nrecs, 1),
                nt
            )

            # compute forward
            swforward!(wavesim, matprop, [Shot(srcs, recs)])
            # save seismograms
            res = copy(recs.seismograms)

            # same sources, but positions reversed
            reverse!(srcs.positions, dims=1)
            # same receivers, but positions reversed
            reverse!(recs.positions, dims=1)
            # compute forward
            swforward!(wavesim, matprop, [Shot(srcs, recs)])
            
            # test that seismograms are the same
            for i in 1:nrecs
                @test res[:, i] ≈ recs.seismograms[:, nrecs-i+1]
            end
        end

        @testset "Test 2D $(parall) single-source multiple-receivers CPML" begin
            #  velocity setup
            c0 = 1000.0
            c0max = 1300.0
            nx = ny = 801
            r = 150
            vel = gaussian_vel_2D(nx, ny, c0, c0max, r)
            matprop = VpAcousticCDMaterialProperty(vel)
            # numerics
            nt = 700
            dx = dy = 2.5
            lx, ly = (nx-1) * dx, (ny-1) * dy
            dt = dx / c0max / sqrt(2)
            halo = 20
            rcoef = 0.0001
            f0 = 5.0
            # wave simulation
            params = InputParametersAcoustic(nt, dt, [nx, ny], [dx, dy],
                     CPMLBoundaryConditionParameters(halo, rcoef, false))
            wavesim = build_wavesim(params; parall=parall)

            # single source at 10 grid points from top CPML boundary
            times = collect(range(0.0; step=dt, length=nt))
            xsrc = lx / 2
            ysrc = (halo + 10) * dy
            srcs = ScalarSources(
                reshape([xsrc, ysrc], 1, 2),
                reshape(rickersource1D.(times, 2/f0, f0), nt, 1),
                f0
            )
            # multiple receivers at different distances
            nrecs = 4
            dist = 50
            xrecs = (lx/2) .- dist .* ((nrecs+1)/2 .- collect(1:nrecs)) .* dx
            yrec  = ly - (halo + 10) * dy  
            recs = ScalarReceivers(
                hcat(xrecs, fill(yrec, nrecs)),
                nt
            )

            # compute forward
            swforward!(wavesim, matprop, [Shot(srcs, recs)])
            # save seismograms
            res = copy(recs.seismograms)

            # same receivers, but positions reversed
            reverse!(recs.positions, dims=1)
            # compute forward
            swforward!(wavesim, matprop, [Shot(srcs, recs)])
            
            # test that seismograms are the same
            for i in 1:nrecs
                @test res[:, i] ≈ recs.seismograms[:, nrecs-i+1]
            end
        end

        @testset "Test 2D $(parall) multiple-sources multiple-receivers CPML" begin
            #  velocity setup
            c0 = 1000.0
            c0max = 1300.0
            nx = ny = 801
            r = 150
            vel = gaussian_vel_2D(nx, ny, c0, c0max, r)
            matprop = VpAcousticCDMaterialProperty(vel)
            # numerics
            nt = 700
            dx = dy = 2.5
            lx, ly = (nx-1) * dx, (ny-1) * dy
            dt = dx / c0max / sqrt(2)
            halo = 20
            rcoef = 0.0001
            f0 = 5.0
            # wave simulation
            params = InputParametersAcoustic(nt, dt, [nx, ny], [dx, dy],
                     CPMLBoundaryConditionParameters(halo, rcoef, false))
            wavesim = build_wavesim(params; parall=parall)

            # multiple sources at differece distances
            times = collect(range(0.0; step=dt, length=nt))
            nsrcs = 3
            distsrcs = 20
            xsrcs = (lx/2) .- distsrcs .* ((nsrcs+1)/2 .- collect(1:nsrcs)) .* dx
            ysrc  = (halo + 10) * dy
            srcs = ScalarSources(
                hcat(xsrcs, fill(ysrc, nsrcs)),
                repeat(rickersource1D.(times, 2/f0, f0), 1, nsrcs),
                f0
            )
            # multiple receivers at different distances
            nrecs = 4
            distrecs = 50
            xrecs = (lx/2) .- distrecs .* ((nrecs+1)/2 .- collect(1:nrecs)) .* dx
            yrec  = ly - (halo + 10) * dy  
            recs = ScalarReceivers(
                hcat(xrecs, fill(yrec, nrecs)),
                nt
            )

            # compute forward
            swforward!(wavesim, matprop, [Shot(srcs, recs)])
            # save seismograms
            res = copy(recs.seismograms)

            # same sources, but positions reversed
            reverse!(srcs.positions, dims=1)
            # same receivers, but positions reversed
            reverse!(recs.positions, dims=1)
            # compute forward
            swforward!(wavesim, matprop, [Shot(srcs, recs)])
            
            # test that seismograms are the same
            for i in 1:nrecs
                @test res[:, i] ≈ recs.seismograms[:, nrecs-i+1]
            end
        end
    end
end