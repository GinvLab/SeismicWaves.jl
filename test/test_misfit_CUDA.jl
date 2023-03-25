using Test
using SeismicWaves

include("utils.jl")

using Logging
error_logger = ConsoleLogger(stderr, Logging.Error)
with_logger(error_logger) do
    @testset "Test 1D swmisfit!" begin
        # Physics
        c0 = 2000.0
        f0 = 10.0
        # Numerics
        nt = 1000
        nx = 101
        dx = 10.0
        dt = dx / c0
        halo = 20
        rcoef = 0.0001
        params, srcs, recs, vel = setup_constant_vel_1D_CPML(nt, dt, nx, dx, c0, f0, halo, rcoef)

        # Solve gradient without checkpointing
        mis = swmisfit!(params, vel, [srcs => recs]; parall=:GPU)
        @test mis > 0
    end
end
