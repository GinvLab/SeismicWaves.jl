using Test
using DSP, NumericalIntegration, LinearAlgebra
using CUDA: CUDA
using Logging
using SeismicWaves

with_logger(ConsoleLogger(stderr, Logging.Warn)) do
    test_backends = [:serial, :threads]
    # test GPU backend only if CUDA is functional
    if @isdefined(CUDA) && CUDA.functional()
        push!(test_backends, :CUDA)
    end
    if @isdefined(AMDGPU) && AMDGPU.functional()
        push!(test_backends, :AMDGPU)
    end

    for parall in test_backends
        @testset "Test 1D $(parall) single-shot snapshotting" begin
            #  velocity setup
            c0 = 1000.0
            c0max = 1300.0
            nx = 501
            r = 100
            vel = gaussian_vel_1D(nx, c0, c0max, r)
            matprop = VpAcousticCDMaterialProperties(vel)
            # numerics
            nt = 2000
            dx = 2.5
            lx = (nx - 1) * dx
            dt = dx / c0max
            halo = 20
            rcoef = 0.0001
            f0 = 5.0
            snapevery = 50
            # wave simulation
            params = InputParametersAcoustic(nt, dt, (nx,), (dx,),
                CPMLBoundaryConditionParameters(halo, rcoef, false))
            wavesim = build_wavesim(params, matprop; parall=parall, snapevery=snapevery)

            # single source at 100 grid points from CPML boundary
            times = collect(range(0.0; step=dt, length=nt))
            xsrc = (halo + 100) * dx
            srcs = ScalarSources(
                reshape([xsrc], 1, 1),
                reshape(rickerstf.(times, 2 / f0, f0), nt, 1),
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
            snaps = swforward!(wavesim, matprop, [ScalarShot(srcs, recs)])

            @test snaps !== nothing
            @test length(snaps) == 1
            @test sort(keys(snaps[1])) == [i for i in snapevery:snapevery:nt]
            @test all(s -> !(norm(s["pcur"].value) ≈ 0), [s for (it, s) in snaps[1]])
            kk = sort(keys(snaps[1]))
            @test all(it -> !(snaps[1][it[1]]["pcur"].value ≈ snaps[1][it[2]]["pcur"].value),
                [(it1, it2) for (it1, it2) in zip(kk[1:end-1], kk[2:end])])
        end

        @testset "Test 1D $(parall) multi-shot snapshotting" begin
            #  velocity setup
            c0 = 1000.0
            c0max = 1300.0
            nx = 501
            r = 100
            vel = gaussian_vel_1D(nx, c0, c0max, r)
            matprop = VpAcousticCDMaterialProperties(vel)
            # numerics
            nt = 2000
            dx = 2.5
            lx = (nx - 1) * dx
            dt = dx / c0max
            halo = 20
            rcoef = 0.0001
            f0 = 5.0
            snapevery = 50
            # wave simulation
            params = InputParametersAcoustic(nt, dt, (nx,), (dx,),
                CPMLBoundaryConditionParameters(halo, rcoef, false))
            wavesim = build_wavesim(params, matprop; parall=parall, snapevery=snapevery)

            # single source at 100 grid points from CPML boundary
            times = collect(range(0.0; step=dt, length=nt))
            xsrc = (halo + 100) * dx
            srcs = ScalarSources(
                reshape([xsrc], 1, 1),
                reshape(rickerstf.(times, 2 / f0, f0), nt, 1),
                f0
            )
            # multiple receivers at different distances
            nrecs = 4
            dist = 50
            recs = ScalarReceivers(
                reshape(xsrc .+ dist .* collect(1:nrecs) .* fill(dx, nrecs), nrecs, 1),
                nt
            )

            # single source at 50 grid points from CPML boundary
            times = collect(range(0.0; step=dt, length=nt))
            xsrc = (halo + 50) * dx
            srcs2 = ScalarSources(
                reshape([xsrc], 1, 1),
                reshape(rickerstf.(times, 2 / f0, f0), nt, 1),
                f0
            )
            # multiple receivers at different distances
            nrecs = 4
            dist = 50
            recs2 = ScalarReceivers(
                reshape(xsrc .+ dist .* collect(1:nrecs) .* fill(dx, nrecs), nrecs, 1),
                nt
            )

            # compute forward
            snaps = swforward!(wavesim, matprop, [ScalarShot(srcs, recs), ScalarShot(srcs2, recs2)])

            @test snaps !== nothing
            @test length(snaps) == 2
            @test sort(keys(snaps[1])) == sort(keys(snaps[2]))
            @test all(it -> !(snaps[1][it]["pcur"].value ≈ snaps[2][it]["pcur"].value), [it for it in keys(snaps[2])])
        end
    end
end
