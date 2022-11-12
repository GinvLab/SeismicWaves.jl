using Test, ReferenceTests
using BSON

using SeismicWaves: rickersource1D, InpParamAcou
using SeismicWaves.AcousticWaves: oneiter_reflbound!, oneiter_GAUSSTAP!, oneiter_CPML!, oneiter_CPML!slow, initGausboundcon, initCPML

include("utils.jl")

ROOT_REF_FLD = joinpath(@__DIR__, "references", "acoustic2D")
ONEITER_FLD = joinpath(ROOT_REF_FLD, "oneiter")

# simple setup for acoustic2D simulations
function simple_acoustic2D_setup(nt, nx, nz, dh, dt;
                                 vel_const=2000.0, vel_grad=12.0,
                                 f0=12.0, ampl=1000.0,
                                 pad_CPML=nothing)
    # create velocity model
    velmod = zeros(nx,nz)
    for i=1:nx
        for j=1:nz
            velmod[i,j] = vel_const + vel_grad*(j-1)
        end
    end
    # compute maximum value for velocity
    velmax = maximum(velmod)
    # pad left, right and bottom for PML layers
    if pad_CPML !== nothing
        left = repeat(velmod[1:1,:],pad_CPML,1)
        right = repeat(velmod[end:end,:],pad_CPML,1)
        velmod = vcat(left,velmod,right)
        # pad also the bottom...
        bottom = repeat(velmod[:,end:end],1,pad_CPML)
        velmod = hcat(velmod,bottom)
    end
    # factor for loops
    fact = velmod.^2 .* (dt^2/dh^2)
    # only one source
    ijsrcs = [round(Int, size(velmod,1)/2) 3]     # 1x2 Matrix
    # source-time function
    t = collect(Float64,range(0.0,step=dt,length=nt))
    t0 = 1.20 / f0
    sourcetf = ampl .* reshape(rickersource1D(t,t0,f0),:,1)
    # pre-scale source-time function
    dt2srctf = fact[ijsrcs[1,1],ijsrcs[1,2]] .* sourcetf
    return fact, ijsrcs, dt2srctf, velmax
end

@testset "Test reflective boundary" begin
    # folder for storinng reference values
    LOCAL_REF_FLD = joinpath(ONEITER_FLD, "reflbound")

    # numerics
    nt = 50
    nx, nz = 211, 120
    dh, dt = 10.0, 0.0012
    # get simple setup
    fact, ijsrcs, dt2srctf, _ = simple_acoustic2D_setup(nt, nx, nz, dh, dt)
    # allocate pressure matrices
    pold = zeros(nx, nz)
    pcur = zeros(nx, nz)
    pnew = zeros(nx, nz)
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
    fact, ijsrcs, dt2srctf, _ = simple_acoustic2D_setup(nt, nx, nz, dh, dt)
    # allocate pressure matrices
    pold = zeros(nx, nz)
    pcur = zeros(nx, nz)
    pnew = zeros(nx, nz)
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

@testset "Test CPML boundary slow" begin
    # folder for storinng reference values
    LOCAL_REF_FLD = joinpath(ONEITER_FLD, "CPML")

    # numerics
    nt = 50
    nx, nz = 211, 120
    dh, dt = 10.0, 0.0012
    f0     = 12.0
    # get simple setup
    fact, ijsrcs, dt2srctf, velmax = simple_acoustic2D_setup(nt, nx, nz, dh, dt; f0=f0, pad_CPML=34)
    # allocate pressure matrices
    pold = zeros(nx, nz)
    pcur = zeros(nx, nz)
    pnew = zeros(nx, nz)
    # allocate CMPL matrices
    psi_x = zeros(nx,nz)
    psi_z = zeros(nx,nz)
    xi_x  = zeros(nx,nz)
    xi_z  = zeros(nx,nz)
    # setup CMPL BDCs
    cpml = initCPML(InpParamAcou(ntimesteps=nt,nx=nx,nz=nz,dt=dt,dh=dh,
                                 savesnapshot=false,snapevery=0,
                                 boundcond="PML",freeboundtop=true,
                                 infoevery=0,smoothgrad=false),
                    velmax, f0)
    # run iterations
    for t=1:nt
        pold,pcur,pnew = oneiter_CPML!slow(nx,nz,fact,
                                           pnew,pold,pcur,
                                           dt2srctf,
                                           psi_x,psi_z,xi_x,xi_z,
                                           cpml,ijsrcs,t)
    end
    # compare with reference
    @test_reference joinpath(LOCAL_REF_FLD, "simple_CPML_nt$(nt).bson") Dict(:pcur=>pcur) by=comp
end

@testset "Test CPML boundary" begin
    # folder for storinng reference values
    LOCAL_REF_FLD = joinpath(ONEITER_FLD, "CPML")

    # numerics
    nt = 50
    nx, nz = 211, 120
    dh, dt = 10.0, 0.0012
    f0     = 12.0
    # get simple setup
    fact, ijsrcs, dt2srctf, velmax = simple_acoustic2D_setup(nt, nx, nz, dh, dt; f0=f0, pad_CPML=34)
    # allocate pressure matrices
    pold = zeros(nx, nz)
    pcur = zeros(nx, nz)
    pnew = zeros(nx, nz)
    # allocate CMPL matrices
    psi_x = zeros(nx,nz)
    psi_z = zeros(nx,nz)
    xi_x  = zeros(nx,nz)
    xi_z  = zeros(nx,nz)
    # setup CMPL BDCs
    cpml = initCPML(InpParamAcou(ntimesteps=nt,nx=nx,nz=nz,dt=dt,dh=dh,
                                 savesnapshot=false,snapevery=0,
                                 boundcond="PML",freeboundtop=true,
                                 infoevery=0,smoothgrad=false),
                    velmax, f0)
    # run iterations
    for t=1:nt
        pold,pcur,pnew = oneiter_CPML!(nx,nz,fact,
                                       pnew,pold,pcur,
                                       dt2srctf,
                                       psi_x,psi_z,xi_x,xi_z,
                                       cpml,ijsrcs,t)
    end
    # compare with reference
    @test_reference joinpath(LOCAL_REF_FLD, "simple_CPML_nt$(nt).bson") Dict(:pcur=>pcur) by=comp
end