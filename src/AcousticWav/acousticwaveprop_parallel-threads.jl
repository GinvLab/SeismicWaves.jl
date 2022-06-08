
#######################################3

function oneiter_threads_CPML!slow(fact::Array{Float64,2},pnew::Array{Float64,2},
                           pold::Array{Float64,2},pcur::Array{Float64,2},#dt2srctf::Array{Float64,2},
                           #dpdx::Array{Float64,2},dpdz::Array{Float64,2},
                           #d2pdx2::Array{Float64,2},d2pdz2::Array{Float64,2},
                           psi_x::Array{Float64,2},psi_z::Array{Float64,2},
                           xi_x::Array{Float64,2},xi_z::Array{Float64,2},cpml::CoefPML)
                          


    ##-------------------------------------------
    ###############################
    ### SLOW but simple VERSION
    ###############################

    nx,nz = size(pcur)
    
    ## compute current psi_x and psi_z first (need derivatives next)
    @inbounds  Threads.@threads for j = 2:nz-1 # 3:nz-2
        @inbounds for i = 2:nx-1 # 3:nx-2
            dpdx = pcur[i+1,j]-pcur[i,j] 
            dpdz = pcur[i,j+1]-pcur[i,j]
            psi_x[i,j] = cpml.b_x_half[i] / cpml.K_x_half[i] * psi_x[i,j] + cpml.a_x_half[i] * dpdx
            psi_z[i,j] = cpml.b_z_half[j] / cpml.K_z_half[j] * psi_z[i,j] + cpml.a_z_half[j] * dpdz
        end
    end

    @inbounds  Threads.@threads for j = 2:nz-1 # 3:nz-2
        @inbounds for i = 2:nx-1 # 3:nx-2

            d2pdx2 = pcur[i+1,j]-2.0*pcur[i,j]+pcur[i-1,j]            
            d2pdz2 = pcur[i,j+1]-2.0*pcur[i,j]+pcur[i,j-1]

            dpsidx = psi_x[i,j] - psi_x[i-1,j] 
            dpsidz = psi_z[i,j] - psi_z[i,j-1]

            xi_x[i,j] = cpml.b_x[i] / cpml.K_x_half[i] * xi_x[i,j] + cpml.a_x[i] * (d2pdx2 + dpsidx)
            xi_z[i,j] = cpml.b_z[j] / cpml.K_z_half[j] * xi_z[i,j] + cpml.a_z[j] * (d2pdz2 + dpsidz)
            
            damp = fact[i,j] * (dpsidx + dpsidz + xi_x[i,j] + xi_z[i,j])

            # update pressure
            pnew[i,j] = 2.0*pcur[i,j] -pold[i,j] +
                fact[i,j]*(d2pdx2 + d2pdz2) + damp
            
        end
    end

   return #pold,pcur,pnew
end

##=======================================================================##

"""
  Calculate one iteration - one time step - for the acoustic wave 
   equation with CPML absorbing boundary conditions
"""
function oneiter_threads_CPML!(fact::Array{Float64,2},pnew::Array{Float64,2},
                               pold::Array{Float64,2},pcur::Array{Float64,2},
                               #dpdx::Array{Float64,2},dpdz::Array{Float64,2},
                               #d2pdx2::Array{Float64,2},d2pdz2::Array{Float64,2},
                               psi_x::Array{Float64,2},psi_z::Array{Float64,2},
                               xi_x::Array{Float64,2},xi_z::Array{Float64,2},cpml::CoefPML)
                               

    # Komatitsch, D., and Martin, R., 2007, An unsplit convolutional perfectly
    # matched layer improved at grazing incidence for the seismic wave
    # equation: Geophysics, 72, SM155–SM167. doi:10.1190/1.2757586
    #       
    # Pasalic, D., and McGarry, R., 2010, Convolutional perfectly matched layer
    # for isotropic and anisotropic acoustic wave equations: 80th Annual
    # International Meeting, SEG, Expanded Abstracts, 2925–2929.

    nx,nz = size(pcur)

    ##====================================================
    ## Loop only withing PML layers to spare calculating zeros...
    # ipmlidxs = [2, nptspml_x, nx-nptspml_x+1, nx-1]
    # jpmlidxs = [2, nptspml_z, nz-nptspml_z+1, nz-1]
    ##====================================================

    #println("$(myid()), $(saslice.irange), $(saslice.jrange), $(saslice.ipmlrange), $(saslice.jpmlrange) ")
    #println("$(myid()), $(saslice.irangeinternal), $(saslice.jrangeinternal)")
    
    ##====================================================
    ## Calculate PML stuff only on the borders...
    
    # @show cpml.ipmlidxs[2]+1:cpml.ipmlidxs[3]-1
    # @show cpml.jpmlidxs[2]+1:cpml.jpmlidxs[3]-1
    
    irange = vcat(cpml.ipmlidxs[1]:cpml.ipmlidxs[2],cpml.ipmlidxs[3]:cpml.ipmlidxs[4])
    jrange = vcat(cpml.jpmlidxs[1]:cpml.jpmlidxs[2],cpml.jpmlidxs[3]:cpml.jpmlidxs[4])

    ## X
    # for j = 1:nz # 1:nz !!
    #     for ii=(1,3)
    #         for i = cpml.ipmlidxs[ii]:cpml.ipmlidxs[ii+1]
    Threads.@threads for j = 1:nz # 1:nz !!
        for i in irange
            dpdx = pcur[i+1,j]-pcur[i,j] 
            psi_x[i,j] = cpml.b_x_half[i] / cpml.K_x_half[i] * psi_x[i,j] + cpml.a_x_half[i] * dpdx
        end
    end
    
    ## Z
    # for jj=(1,3)
    #    for j = cpml.jpmlidxs[jj]:cpml.jpmlidxs[jj+1]
    #         for i = 1:nx # 1:nx !!
    Threads.@threads for i = 1:nx # 1:nx !!
        for j in jrange
            dpdz = pcur[i,j+1]-pcur[i,j]  
            psi_z[i,j] = cpml.b_z_half[j] / cpml.K_z_half[j] * psi_z[i,j] + cpml.a_z_half[j] * dpdz
        end
    end    

    
    ##====================================================
    ## Calculate PML stuff only on the boundaries...
    ## X borders
    # for j = 2:nz-1 # 2:nz-1 !!
    #        for ii=(1,3)
    #            for i = cpml.ipmlidxs[ii]:cpml.ipmlidxs[ii+1]
    Threads.@threads  for j = 2:nz-1 # 2:nz-1 !!
        for i in irange                
            
            d2pdx2 = pcur[i+1,j]-2.0*pcur[i,j]+pcur[i-1,j]            
            d2pdz2 = pcur[i,j+1]-2.0*pcur[i,j]+pcur[i,j-1]
            dpsidx = psi_x[i,j] - psi_x[i-1,j] 
            dpsidz = psi_z[i,j] - psi_z[i,j-1] 
            
            xi_x[i,j] = cpml.b_x[i] / cpml.K_x_half[i] * xi_x[i,j] + cpml.a_x[i] * (d2pdx2 + dpsidx)
            xi_z[i,j] = cpml.b_z[j] / cpml.K_z_half[j] * xi_z[i,j] + cpml.a_z[j] * (d2pdz2 + dpsidz)
            
            damp = fact[i,j] * (dpsidx + dpsidz + xi_x[i,j] + xi_z[i,j])

            # update pressure
            pnew[i,j] = 2.0*pcur[i,j] -pold[i,j] + fact[i,j]*(d2pdx2 + d2pdz2) + damp
        end
    end
    

    ## Calculate PML stuff only on the boundaries...
    ## Z bouers
    # for jj=(1,3)
    #     Threads.@threads for j = cpml.jpmlidxs[jj]:cpml.jpmlidxs[jj+1]
    Threads.@threads for j in jrange
        ##--------------------------------------------------------------------------
        ## EXCLUDE CORNERS, because already visited in the previous X-boundaries loop!
        ##  (It would lead to wrong accumulation of pnew[i,j], etc. otherwise...)
        ##   for i = [...]+1 : [...]-1
        ##--------------------------------------------------------------------------
        for i = cpml.ipmlidxs[2]+1:cpml.ipmlidxs[3]-1               

            d2pdx2 = pcur[i+1,j]-2.0*pcur[i,j]+pcur[i-1,j]            
            d2pdz2 = pcur[i,j+1]-2.0*pcur[i,j]+pcur[i,j-1]
            dpsidx = psi_x[i,j] - psi_x[i-1,j] 
            dpsidz = psi_z[i,j] - psi_z[i,j-1] 
            
            xi_x[i,j] = cpml.b_x[i] / cpml.K_x_half[i] * xi_x[i,j] + cpml.a_x[i] * (d2pdx2 + dpsidx)
            xi_z[i,j] = cpml.b_z[j] / cpml.K_z_half[j] * xi_z[i,j] + cpml.a_z[j] * (d2pdz2 + dpsidz)
            
            damp = fact[i,j] * (dpsidx + dpsidz + xi_x[i,j] + xi_z[i,j])

            # update pressure
            pnew[i,j] = 2.0*pcur[i,j] -pold[i,j] + fact[i,j]*(d2pdx2 + d2pdz2) + damp

        end
    end


    ##----------------------------------------------------------
    ## Calculate stuff in the INTERNAL part of the model
    Threads.@threads for j = cpml.jpmlidxs[2]+1:cpml.jpmlidxs[3]-1    #2:nz-1 
        for i = cpml.ipmlidxs[2]+1:cpml.ipmlidxs[3]-1   #2:nx-1 
            
            d2pdx2 = pcur[i+1,j]-2.0*pcur[i,j]+pcur[i-1,j]            
            d2pdz2 = pcur[i,j+1]-2.0*pcur[i,j]+pcur[i,j-1]
            
            pnew[i,j] = 2.0*pcur[i,j] -pold[i,j] + fact[i,j]*(d2pdx2 + d2pdz2)
        end
    end

    return #pold,pcur,pnew
end

#######################################################################################3

"""
  Solver for 2D acoustic wave equation (parameters: velocity only). 
  Threaded version.
 """
function solveacoustic2D_parathreads(inpar::InpParamAcou,ijsrcs::Array{Array{Int64,2},1},
                                    vel::Array{Float64,2}, ijrecs::Array{Array{Int64,2},1},
                                    sourcetf::Array{Array{Float64,2},1}, srcdomfreq::Array{Float64,1} )

    ##
    ## The second-order staggered-grid formulation of Madariaga (1976) and Virieux (1986) is used:
    ##
    ##          p          dp/dx       p   
    ##            +---------+---------+ ---> x  
    ##            |         |         |
    ##            |         |         |
    ##            |         |         |
    ##            |         |         |
    ##            |         |         |
    ##      dp/dz +---------+         |
    ##            |                   |
    ##            |                   |
    ##            |                   |
    ##            |                   |
    ##            |                   |
    ##            +-------------------+  
    ##           p                     p
    ##            |
    ##            |
    ##            \/ z
    ##

    @assert length(sourcetf)==length(ijsrcs)
    @assert length(sourcetf)==length(ijrecs)
    @assert length(sourcetf)==length(srcdomfreq)
    @assert all(vel.>0.0)
    
    dh = inpar.dh
    nx = inpar.nx #nxy[1] 
    nz = inpar.nz #nxy[2] 
    dt = inpar.dt
    nshots = length(ijsrcs)
    if verbose>0
        @show length(ijsrcs)
    end
 
    ## Check Courant condition
    vel_max = maximum(vel)
    Cou = vel_max*dt*sqrt(1/dh^2+1/dh^2)
    if verbose>0
        println("Courant number: ",Cou)
    end
    @assert  Cou <= 1.0

    if verbose>0
        println(" Absorbing boundaries: ",inpar.boundcond)
        println(" Free surface condition: ",inpar.freeboundtop)
    end
    
    ## Arrays to export snapshots
    if inpar.savesnapshot==true 
        ntsave = div(inpar.ntimesteps,inpar.snapevery)
        psave = zeros(inpar.nx,inpar.nz,ntsave,nshots)
    end

    ## Arrays to return seismograms
    nrecs = size(ijrecs,1)
    receiv = Array{Array{Float64,2},1}(undef,nshots)
    
    ##=====================
    @show Threads.nthreads()
    
    ## factor for loops
    fact = vel.^2 .* (dt^2/dh^2) # init locally...

    # PML arrays
    # dpdx  = zeros(nx,nz)# zeros(nx,nz)
    # dpdz  = zeros(nx,nz) # zeros(nx,nz)
    # d2pdx2 =zeros(nx,nz) # zeros(nx,nz)
    # d2pdz2 =zeros(nx,nz) # zeros(nx,nz)
    psi_x  =zeros(nx,nz) # zeros(nx,nz)
    psi_z  =zeros(nx,nz) # zeros(nx,nz)
    xi_x   =zeros(nx,nz) # zeros(nx,nz)
    xi_z   =zeros(nx,nz) # zeros(nx,nz)
    
    # memory arrays
    pold =zeros(nx,nz) # zeros(nx,nz)
    pcur =zeros(nx,nz) # zeros(nx,nz)
    pnew =zeros(nx,nz) # zeros(nx,nz)

    ##############################
    #   Loop on shots
    ##############################
    for s=1:nshots

        @assert size(ijsrcs[s],1)==size(sourcetf[s],2)
        ## ensure at least 10 pts per wavelengh  ????
        @assert dh <= vel_max/(10.0 * srcdomfreq[s])

        ##############################
        #   Parameters CPML
        ##############################
        if inpar.boundcond=="PML"
            
            f0 = srcdomfreq[s]
            cpml = initCPML(inpar,vel_max,f0)
                   
            ##-----------------------------------------------------------
            ## Check that source and receivers are inside the PML layers
            for i=1:size(ijsrcs[s],1)
                if cpml.nptspml_x>=ijsrcs[s][i,1]>(nx-cpml.nptspml_x)
                    error("ijsrcs[$(s)][$(i),1] inside PML layers along x")
                end
                if cpml.nptspml_z>=ijsrcs[s][i,2]>(nz-cpml.nptspml_z)
                    error("ijsrcs[$(s)][$(i),1] inside PML layers along z")
                end
            end
            for i=1:size(ijrecs[s],1)
                if cpml.nptspml_x>=ijrecs[s][i,1]>(nx-cpml.nptspml_x)
                    error("ijrecs[$(s)][$(i),1] inside PML layers along x")
                end
                if !inpar.freeboundtop && cpml.nptspml_z>=ijrecs[s][i,2]
                    error("ijrecs[$(s)][$(i),1] inside PML layers along z")
                elseif ijrecs[s][i,2]>(nz-cpml.nptspml_z)
                    error("ijrecs[$(s)][$(i),1] inside PML layers along z")
                end
            end

            ##################################
            # PML arrays
            # Arrays with size of PML areas would be sufficient and save memory,
            #   however allocating arrays with same size than model simplifies
            #   the code in the loops
            # Zeroed at every shot
            # dpdx[:,:] .= 0.0
            # dpdz[:,:] .= 0.0
            # d2pdx2[:,:] .= 0.0
            # d2pdz2[:,:] .= 0.0
            psi_x[:,:] .= 0.0
            psi_z[:,:] .= 0.0
            xi_x[:,:] .= 0.0
            xi_z[:,:] .= 0.0

        elseif inpar.boundcond=="GauTap"

            ## Gaussian taper boundary condition
            gaubc = initGausboundcon()

        end

        ##################################
        # memory arrays
        pold[:,:] .= 0.0
        pcur[:,:] .= 0.0
        pnew[:,:] .= 0.0

        ## seismograms
        nrecs = size(ijrecs[s],1)
        receiv[s] = zeros(Float64,inpar.ntimesteps,nrecs)

        ## pre-scale source time function
##----------------
        dt2srctf = copy(sourcetf[s])  
        for isr=1:size(sourcetf[s],2)
            # Each srctf has to be scaled with the velocity at
            #  same coordinates, etc.: vel^2*(dt^2/dh^2)
            dt2srctf[:,isr] .= fact[ijsrcs[s][isr,1],ijsrcs[s][isr,2]] .* dt2srctf[:,isr]
        end
##----------------

        #####################
        ##  Time loop
        #####################
        if verbose>0
            t1=time()
        end

        isnap = 0
        for t=1:inpar.ntimesteps
            if verbose>0
                t%inpar.infoevery==0 && print("\rShot ",s," t: ",t," of ",inpar.ntimesteps)
            end
            
            ##==================================##
            ##           pressure
            ##==================================##

            if inpar.boundcond=="PML"

                oneiter_threads_CPML!(fact,pnew,pold,pcur,#dpdx,dpdz,#d2pdx2,d2pdz2,
                                      psi_x,psi_z,xi_x,xi_z,cpml)

                
                ##====================================================
                
                # inject source(s)
                @inbounds for l=1:size(dt2srctf,2)
                    isrc = ijsrcs[s][l,1]
                    jsrc = ijsrcs[s][l,2]
                    pnew[isrc,jsrc] = pnew[isrc,jsrc] + dt2srctf[t,l]
                end
                
                # assign the new pold and pcur
                ## https://discourse.julialang.org/t/swap-array-contents/7774/7
                ## kind of swapping array pointers, so NEED TO RETURN NEW BINDINGS,
                ##  otherwise the exchange is lost!
                pold,pcur,pnew = pcur,pnew,pold

            else
                error("Only PML boundary conditions implemented for SharedArrays")
                
            end
 
            ##========================
            ##### receivers
            for r=1:nrecs
                ir = ijrecs[s][r,1]
                jr = ijrecs[s][r,2]
                receiv[s][t,r] = pcur[ir,jr]
            end            


            ##========================
            ##### snapshots
            if (inpar.savesnapshot==true) & (t%inpar.snapevery==0) 
                isnap+=1
                psave[:,:,isnap,s] = pcur 
            end
       
        end ##------- end time loop --------------

        if verbose>0
            t2=time()
            println("\nFWD Time loop: ",t2-t1,"\n")
        end

#end  ##<<<<<<<==================================<<<<<<<<<
        
    end ##------- for ishot=1:... --------------

    if inpar.savesnapshot==true
        return receiv,psave
    else 
        return receiv
    end
end


# ###################======================================##################
# """
#   Solver for computing the gradient of the misfit function for the acoustic 
#    wave equation using the adjoint state method
# """
# function gradacoustic2D_parashared(inpar::InpParamAcou, obsrecv::Array{Array{Float64,2},1},
#                                    invCovds::Union{Vector{Matrix{Float64}},Vector{Diagonal{Float64}}}, #recstd::Float64,
#                                    ijsrcs::Array{Array{Int64,2},1},
#                                    vel::Array{Float64,2}, ijrecs::Array{Array{Int64,2},1},
#                                    sourcetf::Array{Array{Float64,2},1}, srcdomfreq::Array{Float64,1} )
  
#     ##
#     ## The second-order staggered-grid formulation of Madariaga (1976) and Virieux (1986) is used:
#     ##
#     ##          p          dp/dx       p   
#     ##            +---------+---------+ ---> x  
#     ##            |         |         |
#     ##            |         |         |
#     ##            |         |         |
#     ##            |         |         |
#     ##            |         |         |
#     ##      dp/dz +---------+         |
#     ##            |                   |
#     ##            |                   |
#     ##            |                   |
#     ##            |                   |
#     ##            |                   |
#     ##            +-------------------+  
#     ##           p                     p
#     ##            |
#     ##            |
#     ##            \/ z
#     ##

#     ## Bunks et al., 1995 Geophysics, Multiscale seismic waveform inversion.

#     if verbose>0
#         tstart=time()
#     end

#     if inpar.boundcond != "PML"
#         error("gradadj_acoustic2D(): Boundary contitions must be PML for gradient computations.")
#     end


#     @assert length(sourcetf)==length(ijsrcs)
#     @assert length(sourcetf)==length(ijrecs)
#     @assert length(sourcetf)==length(srcdomfreq)
#     @assert all(vel.>0.0)

#     dh = inpar.dh
#     nx = inpar.nx #nxy[1] 
#     nz = inpar.nz #nxy[2] 
#     dt = inpar.dt
#     nt = inpar.ntimesteps
#     nshots = length(ijsrcs)
#     if verbose>0
#         @show length(ijsrcs)
#     end
    
#     ## Check Courant condition
#     vel_max = maximum(vel)
#     Cou = vel_max*dt*sqrt(1/dh^2+1/dh^2)
#     if verbose>0
#         @show Cou
#     end
#     @assert Cou <= 1.0
 

#     ## Check memory requirements to store fwd field
#     totmem = nx*nz*nt*8/1000/1000/1000  ## GB
#     maxMEM = 8.0
#     if totmem>maxMEM
#         println(" Requested mem for pfwdsave: ",totmem," GB, max. allowed ",maxMEM)
#         println(" To allow higher threshold change the parameter 'maxMEM' in function gradacoustic2D_serial()")
#         error("Required memory exceeds set threshold.")
#         return
#     else
#         if verbose>0
#             println(" Requested mem for pfwdsave: ",totmem," GB")
#         end
#     end

    
#     ## Arrays to export snapshots
#     ## nt+1 for the fin. diff. derivative w.r.t time in the adjoint !!
#     pfwdsave = zeros(inpar.nx,inpar.nz,inpar.ntimesteps+2)

#     ## Arrays to return seismograms
#     nrecs = size(ijrecs,1)
#     receiv = Array{Array{Float64,2},1}(undef,nshots)
#     residuals = Array{Array{Float64,2},1}(undef,nshots)
    
#     ## factor for loops
#     fact = vel.^2 .* (dt^2/dh^2)
#     dt2 = dt^2
#     vel3 = vel.^3

#     ## factor for source time functions
#     vel2dt2 = vel.^2 .* dt2

#     # PML arrays
#     # Arrays with size of PML areas would be sufficient and save memory,
#     #  however allocating arrays with same size than model simplifies
#     #  the code in the loops
#     dpdx = zeros(nx,nz)
#     dpdz = zeros(nx,nz)
#     d2pdx2 = zeros(nx,nz)
#     d2pdz2 = zeros(nx,nz)
#     psi_x = zeros(nx,nz)
#     psi_z = zeros(nx,nz)
#     xi_x = zeros(nx,nz)
#     xi_z = zeros(nx,nz)
    
#     # memory arrays
#     pold = zeros(nx,nz)
#     pcur = zeros(nx,nz)
#     pnew = zeros(nx,nz)
#     pveryold = zeros(nx,nz)

#     ## adjoint arrays
#     adjold = zeros(nx,nz)
#     adjcur = zeros(nx,nz)
#     adjnew = zeros(nx,nz)

#     # init gradient
#     curgrad = zeros(nx,nz)
#     grad = zeros(nx,nz)
#     #dpcur2dt2 = zeros(nx,nz)
    
#     ## tmp arrays 
#     tmpdifcalobs = zeros(inpar.ntimesteps)
#     tmpresid = zeros(inpar.ntimesteps)

#     if verbose>0
#         t0=time()
#     end
#     ##############################
#     #   Loop on shots
#     ##############################
#     for s=1:nshots

#         @assert size(ijsrcs[s],1)==size(sourcetf[s],2)
#         ## ensure at least 10 pts per wavelengh ????
#         @assert dh <= vel_max/(10.0 * srcdomfreq[s])
        
#         ##############################
#         #   Parameters CPML
#         ##############################
#         f0 = srcdomfreq[s]
#         cpml = initCPML(inpar,vel_max,f0)

#         ##-----------------------------------------------------------
#         ## Check that source and receivers are inside the PML layers
#         for i=1:size(ijsrcs[s],1)
#             if cpml.nptspml_x>=ijsrcs[s][i,1]>(nx-cpml.nptspml_x)
#                 error("ijsrcs[$(s)][$(i),1] inside PML layers along x")
#             end
#             if cpml.nptspml_z>=ijsrcs[s][i,2]>(nz-cpml.nptspml_z)
#                 error("ijsrcs[$(s)][$(i),1] inside PML layers along z")
#             end
#         end
#         for i=1:size(ijrecs[s],1)
#             if cpml.nptspml_x>=ijrecs[s][i,1]>(nx-cpml.nptspml_x)
#                 error("ijrecs[$(s)][$(i),1] inside PML layers along x")
#             end
#             if !inpar.freeboundtop && cpml.nptspml_z>=ijrecs[s][i,2]
#                 error("ijrecs[$(s)][$(i),1] inside PML layers along z")
#             elseif ijrecs[s][i,2]>(nz-cpml.nptspml_z)
#                 error("ijrecs[$(s)][$(i),1] inside PML layers along z")
#             end
#         end
        
#         ##################################
#         # PML arrays
#         # Arrays with size of PML areas would be sufficient and save memory,
#         #   however allocating arrays with same size than model simplifies
#         #   the code in the loops
#         # Zeroed at every shot
#         dpdx[:,:] .= 0.0
#         dpdz[:,:] .= 0.0 
#         d2pdx2[:,:] .= 0.0
#         d2pdz2[:,:] .= 0.0
#         psi_x[:,:] .= 0.0
#         psi_z[:,:] .= 0.0
#         xi_x[:,:] .= 0.0
#         xi_z[:,:] .= 0.0

#         ##################################
#         # memory arrays
#         pold[:,:] .= 0.0
#         pcur[:,:] .= 0.0
#         pnew[:,:] .= 0.0

#         ##################################
#         ## seismograms
#         nrecs = size(ijrecs[s],1)
#         receiv[s] = zeros(Float64,inpar.ntimesteps,nrecs)
#         ## residuals for adjoint
#         ##residuals[s] = zeros(inpar.ntimesteps,nrecs)

#         ## Pre-scale source time function
#         ##    ??? add 2 rows of zeros to match time step adjoint (0 and nt+1)
#         dt2srctf = [sourcetf[s][:,:]; zeros(size(sourcetf[s],2))]
# ##----------------
#         for isr=1:size(sourcetf[s],2)
#             # Each srctf has to be scaled with the velocity at
#             #  same coordinates, etc.: vel^2*(dt^2/dh^2)
#             dt2srctf[:,isr] .= fact[ijsrcs[s][isr,1],ijsrcs[s][isr,2]] .* dt2srctf[:,isr]
#         end
# ##----------------

#         # current sources for forward and adjoint calculations
#         thishotijsrcs_fwd = ijsrcs[s]
#         thishotijsrcs_adj = ijrecs[s]

#         ntobs = size(obsrecv[s],1)
#         ### ntobs+1 to mach the adjoint field time step
#         thishotsrctf_adj = Array{Float64}(undef,ntobs+1,nrecs)
#         thishotsrctf_adj[end,:] .= 0.0 ### ntobs+1 to mach the adjoint field time step

#         ######################################
#         ##      residuals calculation       ##
#         ######################################
#         #isnap = 0
#         if verbose>0
#             t1=time()
#             println(" t1-t0: $(t1-t0)")
#         end
#         ## One more time step!
#         ## nt+1 for the fin. diff. derivative w.r.t time in the adjoint !!
#         @assert (size(dt2srctf,1)>=inpar.ntimesteps + 1  )

#         for t=1:inpar.ntimesteps + 1 ## + 1 !!
#             if verbose>0
#                 t%inpar.infoevery==0 && print("\rShot ",s," t: ",t," of ",inpar.ntimesteps)
#             end
            
#             ##==================================##
#             ##           pressure
#             ##==================================##

#             ### arrays are swapped before being returned from oneiter_CPML!(),
#             ###   that's why we need to return them (to make that happen)
#             if useslowfd
#                 pold,pcur,pnew = oneiter_CPML!slow(nx,nz,fact,pnew,pold,pcur,dt2srctf,
#                                                    dpdx,dpdz,d2pdx2,d2pdz2,
#                                                    psi_x,psi_z,xi_x,xi_z,
#                                                    cpml,thishotijsrcs_fwd,t)
#             else
#                 pold,pcur,pnew = oneiter_CPML!(nx,nz,fact,pnew,pold,pcur,dt2srctf,
#                                                dpdx,dpdz,d2pdx2,d2pdz2,
#                                                psi_x,psi_z,xi_x,xi_z,
#                                                cpml,thishotijsrcs_fwd,t)
#             end
            
#             ##========================
#             ##### receivers
#             ## skip the additional time step for receivers
#             if t<=inpar.ntimesteps 
#                 @inbounds for r=1:nrecs
#                     ir = ijrecs[s][r,1]
#                     jr = ijrecs[s][r,2]
#                     receiv[s][t,r] = pcur[ir,jr]
#                 end            
#             end
            
#             ##========================
#             ## save forward run
#             ## t+1 because the first is zeros in the past, for adjoint
#             @inbounds pfwdsave[:,:,t+1] .= pcur

#        end ##------- end time loop --------------

#         if verbose>0
#             t2=time()
#         end

#         ##============= ===========================
#         ###------- Residuals -------------
#         ## ddf = obsrecv - receiv
#         ##residuals =  invC_d_onesrc * ddf
#         @inbounds for r=1:nrecs
#             ##OLD: residuals[s][:,r] .= (receiv[s][:,r].-obsrecv[s][:,r]) ./ recstd.^2

#             tmpdifcalobs .= receiv[s][:,r].-obsrecv[s][:,r]
#             ## The followind line works but it allocates
#             ##  residuals[s][:,r] .= invCovds[s] * tmpdifcalobs
#             ## The next line produces zeros as output?!? Why??
#             ##  mul!(residuals[s][:,r], invCovds[s], tmpdifcalobs)
#             ## So using a second temporary array "tmpresid" to hold the
#             ##    results and still avoid allocating...
#             mul!(tmpresid, invCovds[s], tmpdifcalobs)
#             ##residuals[s][:,r] .= tmpresid

#             ## Source time function for adjoint
#             ## REVERSE residuals in time
#             ## last row of thishotsrctfresid must be already zero!
#             #thishotsrctfresid[1:end-1,:] .= residuals[s][end:-1:1,r]
#             thishotsrctf_adj[1:end-1,r] .= tmpresid[end:-1:1] 
#         end    
  
# ##----------------
#         for isr=1:size(thishotsrctf_adj,2)
#             ## The adjoint source is *scaled only* by vel^2*dt^2 instead of vel^2*(dt^2/dh^2)
#             thishotsrctf_adj[:,isr] .= thishotsrctf_adj[:,isr] .* vel2dt2[thishotijsrcs_adj[isr,1],thishotijsrcs_adj[isr,2]]
#         end
# ##----------------

#         if verbose>0
#             t3=time()
#             println("\n Forward calculation Time loop: ",t2-t1)
#             println(" Residuals calculation Time loop: ",t3-t2)
#             println(" Total residuals calculation Time loop: ",t3-t1)
#         end

#         ######################################
#         ##      adjoint calculation         ##
#         ######################################

#         ## adjoint arrays
#         adjold[:,:] .= 0.0
#         adjcur[:,:] .= 0.0
#         adjnew[:,:] .= 0.0

#         # gradient for 1 shot
#         curgrad .= 0.0

#         ## PML arrays
#         dpdx[:,:] .= 0.0
#         dpdz[:,:] .= 0.0
#         d2pdx2[:,:] .= 0.0
#         d2pdz2[:,:] .= 0.0
#         psi_x[:,:] .= 0.0
#         psi_z[:,:] .= 0.0
#         xi_x[:,:] .= 0.0
#         xi_z[:,:] .= 0.0

#         ##==================================##
#         ## time loop
#         if verbose>0
#             t4=time()
#         end
#         nt = inpar.ntimesteps
        
#         ## Adjoint actually going backward in time
#         @assert (size(thishotsrctf_adj,1)>=nt-1 )
#         for t = 1:nt
#             if verbose>0
#                 t%inpar.infoevery==0 && print("\rShot ",s," t: ",t," of ",inpar.ntimesteps)
#                 t5 = time()
#             end
                         
#             ##==================================##
#             ##           adjoint
#             ##==================================##            
#             if useslowfd
#                 adjold,adjcur,adjnew = oneiter_CPML!slow(nx,nz,fact,adjnew,adjold,
#                                                          adjcur,thishotsrctf_adj,
#                                                          dpdx,dpdz,d2pdx2,d2pdz2,
#                                                          psi_x,psi_z,xi_x,xi_z,
#                                                          cpml,thishotijsrcs_adj,t)
                
#             else
#                 adjold,adjcur,adjnew = oneiter_CPML!(nx,nz,fact,adjnew,adjold,
#                                                      adjcur,thishotsrctf_adj,
#                                                      dpdx,dpdz,d2pdx2,d2pdz2,
#                                                      psi_x,psi_z,xi_x,xi_z,
#                                                      cpml,thishotijsrcs_adj,t)
#             end
                        
#             if verbose>1
#                 t6=time()
#                 println("\n Adjoint calculations: ",t6-t5)
#             end

#             ##==================================##
#             ##          correlate
#             ##==================================##
#             ## p is shifted into future, so pcur is p at t+1
#             ##dpcur2dt2[:,:] .=  (pcur .- 2.0.*pold .+ pveryold) ./ dt2
#             @inbounds for j=1:nz
#                 @inbounds for i=1:nx
#                     dpcur2dt2 = (pfwdsave[i,j,nt-t+1] - 2.0 * pfwdsave[i,j,nt-t+2] +
#                                  pfwdsave[i,j,nt-t+3]) / dt2
#                     ## sum in time!
#                     ## pointwise multiplication, integration in time...
#                     curgrad[i,j] = curgrad[i,j] + (adjcur[i,j] * dpcur2dt2)
#                 end
#             end 

#             if verbose>1
#                 t7=time()
#                 println(" Correlating: ",t7-t6)
#             end   

#         end ##------- end time loop --------------

#         ## tot gradient
#         grad .= grad .+ curgrad

#         if verbose>0
#             t9=time()
#             println(" Total adjoint solver time for 1 shot: ",t9-t4)
#         end
        
#     end ##------- for ishot=1:... --------------
#     ##===========================================================##

#     ## scale gradient
#     grad .= (2.0 ./ vel3) .* grad

#     if verbose>0
#         t10 = time()
#         println(" Init etc. : ",t0-tstart)
        
#         println(" Total gradient calculation time for 1 shot: ",t10-tstart)
#     end
        
#     return grad ##,residuals
# end


# ##########################################################################
# ##########################################################################
