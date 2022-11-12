using Test, ReferenceTests
using BSON

using SeismicWaves: rickersource1D
using SeismicWaves.AcousticWaves: oneiter_reflbound!, oneiter_GAUSSTAP!, initGausboundcon

include("utils.jl")

ROOT_REF_FLD = joinpath(@__DIR__, "references", "acoustic2D")
ONEITER_FLD = joinpath(ROOT_REF_FLD, "oneiter")

# simple setup for acoustic2D simulations
function simple_setup(nt, nx, nz, dh, dt; vel_const=2000.0, vel_grad=12.0)
    # create velocity model
    velmod = zeros(nx,nz)
    for i=1:nx
        for j=1:nz
            velmod[i,j] = vel_const + vel_grad*(j-1)
        end
    end
    # factor for loops
    fact = velmod.^2 .* (dt^2/dh^2)
    # only one source position = (nx/2, 3)
    ijsrcs = [round(Int, nx/2) 3]     # 1x2 Matrix
    # source-time function
    t = collect(Float64,range(0.0,step=dt,length=nt))
    f0 = 12.0
    t0 = 1.20 / f0
    sourcetf = 1000.0 .* reshape(rickersource1D(t,t0,f0),:,1)
    # pre-scale source-time function
    dt2srctf = fact[ijsrcs[1,1],ijsrcs[1,2]] .* sourcetf
    return fact, ijsrcs, dt2srctf
end

@testset "Test reflective boundary" begin
    # folder for storinng reference values
    LOCAL_REF_FLD = joinpath(ONEITER_FLD, "reflbound")

    # numerics
    nt = 50
    nx, nz = 211, 120
    dh, dt = 10.0, 0.0012
    # get simple setup
    fact, ijsrcs, dt2srctf = simple_setup(nt, nx, nz, dh, dt)
    # allocate pressure matrices
    pold, pcur, pnew = zeros(nx, nz), zeros(nx, nz), zeros(nx, nz)
    # run iterations
    for t=1:nt
        oneiter_reflbound!(nx,nz,fact,
                           pnew,pold,pcur,
                           dt2srctf,ijsrcs,t)
    end
    # compare with reference
    @test_reference joinpath(LOCAL_REF_FLD, "simple_reflective_boundary_nt$(nt).bson") Dict(:pcur=>pcur) by=comp
end

@testset "Test Gaussian boundary" begin
    # folder for storinng reference values
    LOCAL_REF_FLD = joinpath(ONEITER_FLD, "gausstaper")

    # numerics
    nt = 50
    nx, nz = 211, 120
    dh, dt = 10.0, 0.0012
    # get simple setup
    fact, ijsrcs, dt2srctf = simple_setup(nt, nx, nz, dh, dt)
    # allocate pressure matrices
    pold, pcur, pnew = zeros(nx, nz), zeros(nx, nz), zeros(nx, nz)
    # setup Gaussian taper BDC
    gaubc = initGausboundcon()
    # run iterations
    for t=1:nt
        pold,pcur,pnew = oneiter_GAUSSTAP!(nx,nz,fact,
                                           pnew,pold,pcur,
                                           dt2srctf,ijsrcs,t,gaubc)
    end
    # compare with reference
    @test_reference joinpath(LOCAL_REF_FLD, "simple_gausstaper_boundary_nt$(nt).bson") Dict(:pcur=>pcur) by=comp
end
