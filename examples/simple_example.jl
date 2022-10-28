


using SeismicWaves


###################################################################

function exacouprob()

    ##========================================
    # time stuff
    nt = 1500 
    dt = 0.0012 
    t = collect(Float64,range(0.0,step=dt,length=nt)) # seconds
    #@show dt,(nt-1)*dt

    ##========================================
    # create a velocity model
    nx = 211
    nz = 120
    dh = 10.0 # meters

    velmod = zeros(nx,nz)
    for i=1:nx
        for j=1:nz
            velmod[i,j] = 2000.0 + 12.0*(j-1)
        end
    end

    # pad the lateral borders because there will be some PML layers there
    npad = 34
    left = repeat(velmod[1:1,:],npad,1)
    right = repeat(velmod[end:end,:],npad,1)
    velmod = vcat(left,velmod,right)
    # pad also the bottom...
    bottom = repeat(velmod[:,end:end],1,npad)
    velmod = hcat(velmod,bottom)

    ##========================================
    # source position
    nshots = 6
    nsrc = 1
    ixsrc = round.(Int,LinRange(32,nx-31,nshots))  #round.(Int,[nx/2])
    #ixsrc = round.(Int,[nx/2])
    ijsrcs = Array{Array{Int64,2},1}([zeros(Int64,nsrc,2) for i=1:nshots])
    for i=1:nshots
        ijsrcs[i][:,1] .= ixsrc[i]
        ijsrcs[i][:,2] .= round.(Int,[3 for i=1:nsrc]) #round.(Int,[3 for i=1:nsrc]) #
    end
 
    ##========================================
    ## receiver position
    nrec = 20
    ixrec = round.(Int,LinRange(30,nx-29,nrec))
    ijrecs = Array{Array{Int64,2},1}([zeros(Int,nrec,2) for i=1:nshots])
    for i=1:nshots
        ijrecs[i][:,1] .= ixrec
        ijrecs[i][:,2] .= round.(Int,[3 for i=1:nrec])
    end
          
    ##========================================
    # source-time function
    f0 = 12.0
    t0 = 1.20 / f0
    sourcetf = Array{Array{Float64,2},1}(undef,nshots)
    srcdomfreq = zeros(nshots)
    for i=1:nshots
        ## sourcetime function
        sourcetf[i] = 1000.0 .* reshape(rickersource1D(t,t0,f0),:,1)
        srcdomfreq[i] = f0
    end


    ##============================================
    ## Input parameters for acoustic simulation
    savesnapshot = false
    snapevery = 50
    freeboundtop = true
    boundcond = "PML"  #"GauTap" # "PML"
    println("Boundary conditions: $boundcond ")
    infoevery = 500

    # pressure field in space and time
    inpar = InpParamAcou(ntimesteps=nt,nx=nx,nz=nz,dt=dt,dh=dh,
                         savesnapshot=savesnapshot,snapevery=snapevery,
                         boundcond=boundcond,freeboundtop=freeboundtop,
                         infoevery=infoevery,smoothgrad=true)

    ##===============================================
    ## compute the seismograms
    seism = solveacoustic2D(inpar,ijsrcs,velmod,ijrecs,sourcetf,srcdomfreq,runparallel=true)

    return inpar,velmod,seism
end


##################################################################

