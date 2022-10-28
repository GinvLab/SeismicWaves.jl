
##=======================================================================##

"""
  Solver for 2D acoustic wave equation (parameters: velocity only). 
  Serial version.
"""
function solveacoustic2D_serial(inpar::InpParamAcou,ijsrcs::Array{Array{Int64,2},1},
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
    
    ## factor for loops
    fact = vel.^2 .* (dt^2/dh^2)
    
    # PML arrays
    psi_x = zeros(nx,nz)
    psi_z = zeros(nx,nz)
    xi_x  = zeros(nx,nz)
    xi_z  = zeros(nx,nz)
    
    # memory arrays
    pold = zeros(nx,nz)
    pcur = zeros(nx,nz)
    pnew = zeros(nx,nz)

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

                ### arrays are swapped bofore being returned from oneiter_CPML!()
                if useslowfd
                    pold,pcur,pnew = oneiter_CPML!slow(nx,nz,fact,pnew,pold,pcur,dt2srctf,
                                                       psi_x,psi_z,xi_x,xi_z,
                                                       cpml,ijsrcs[s],t)

                    
                else
                    pold,pcur,pnew = oneiter_CPML!(nx,nz,fact,pnew,pold,pcur,dt2srctf,
                                                   psi_x,psi_z,xi_x,xi_z,
                                                   cpml,ijsrcs[s],t)
                end
                
            elseif inpar.boundcond=="GauTap"
                oneiter_GAUSSTAP!(nx,nz,fact,pnew,pold,pcur,dt2srctf,
                                  ijsrcs[s],t,gaubc,inpar.freeboundtop)
                
            else
                oneiter_reflbound!(nx,nz,fact,pnew,pold,pcur,dt2srctf,
                                   ijsrcs[s],t,inpar.freeboundtop)
                
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

    end ##------- for ishot=1:... --------------

    if inpar.savesnapshot==true
        return receiv,psave
    else 
        return receiv
    end
end


###################======================================##################
"""
  Solver for computing the gradient of the misfit function for the acoustic 
   wave equation using the adjoint state method
"""
function gradacoustic2D_serial(inpar::InpParamAcou, obsrecv::Array{Array{Float64,2},1},
                               invCovds::Union{Vector{Matrix{Float64}},Vector{Diagonal{Float64}}}, #recstd::Float64,
                               ijsrcs::Array{Array{Int64,2},1},
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

    ## Bunks et al., 1995 Geophysics, Multiscale seismic waveform inversion.

    if verbose>0
        tstart=time()
    end

    if inpar.boundcond != "PML"
        error("gradadj_acoustic2D(): Boundary contitions must be PML for gradient computations.")
    end


    @assert length(sourcetf)==length(ijsrcs)
    @assert length(sourcetf)==length(ijrecs)
    @assert length(sourcetf)==length(srcdomfreq)
    @assert all(vel.>0.0)

    dh = inpar.dh
    nx = inpar.nx #nxy[1] 
    nz = inpar.nz #nxy[2] 
    dt = inpar.dt
    nt = inpar.ntimesteps
    nshots = length(ijsrcs)
    if verbose>0
        @show length(ijsrcs)
    end
    
    ## Check Courant condition
    vel_max = maximum(vel)
    Cou = vel_max*dt*sqrt(1/dh^2+1/dh^2)
    if verbose>0
        @show Cou
    end
    @assert Cou <= 1.0
 

    ## Check memory requirements to store fwd field
    totmem = nx*nz*nt*8/1000/1000/1000  ## GB
    maxMEM = 8.0
    if totmem>maxMEM
        println(" Requested mem for pfwdsave: ",totmem," GB, max. allowed ",maxMEM)
        println(" To allow higher threshold change the parameter 'maxMEM' in function gradacoustic2D_serial()")
        error("Required memory exceeds set threshold.")
        return
    else
        if verbose>0
            println(" Requested mem for pfwdsave: ",totmem," GB")
        end
    end

    
    ## Arrays to export snapshots
    ## nt+1 for the fin. diff. derivative w.r.t time in the adjoint !!
    pfwdsave = zeros(inpar.nx,inpar.nz,inpar.ntimesteps+2)

    ## Arrays to return seismograms
    nrecs = size(ijrecs,1)
    receiv = Array{Array{Float64,2},1}(undef,nshots)
    residuals = Array{Array{Float64,2},1}(undef,nshots)
    
    ## factor for loops
    fact = vel.^2 .* (dt^2/dh^2)
    dt2 = dt^2
    vel3 = vel.^3

    ## factor for source time functions
    vel2dt2 = vel.^2 .* dt2

    # PML arrays
    # Arrays with size of PML areas would be sufficient and save memory,
    #  however allocating arrays with same size than model simplifies
    #  the code in the loops
    psi_x = zeros(nx,nz)
    psi_z = zeros(nx,nz)
    xi_x = zeros(nx,nz)
    xi_z = zeros(nx,nz)
    
    # memory arrays
    pold = zeros(nx,nz)
    pcur = zeros(nx,nz)
    pnew = zeros(nx,nz)
    pveryold = zeros(nx,nz)

    ## adjoint arrays
    adjold = zeros(nx,nz)
    adjcur = zeros(nx,nz)
    adjnew = zeros(nx,nz)

    # init gradient
    curgrad = zeros(nx,nz)
    grad = zeros(nx,nz)
    #dpcur2dt2 = zeros(nx,nz)
    
    ## tmp arrays 
    tmpdifcalobs = zeros(inpar.ntimesteps)
    tmpresid = zeros(inpar.ntimesteps)

    if verbose>0
        t0=time()
    end
    ##############################
    #   Loop on shots
    ##############################
    for s=1:nshots

        @assert size(ijsrcs[s],1)==size(sourcetf[s],2)
        ## ensure at least 10 pts per wavelengh ????
        @assert dh <= vel_max/(10.0 * srcdomfreq[s])
        
        ##############################
        #   Parameters CPML
        ##############################
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
        psi_x[:,:] .= 0.0
        psi_z[:,:] .= 0.0
        xi_x[:,:] .= 0.0
        xi_z[:,:] .= 0.0

        ##################################
        # memory arrays
        pold[:,:] .= 0.0
        pcur[:,:] .= 0.0
        pnew[:,:] .= 0.0

        ##################################
        ## seismograms
        nrecs = size(ijrecs[s],1)
        receiv[s] = zeros(Float64,inpar.ntimesteps,nrecs)
        ## residuals for adjoint
        ##residuals[s] = zeros(inpar.ntimesteps,nrecs)

        ## Pre-scale source time function
        ##    ??? add 2 rows of zeros to match time step adjoint (0 and nt+1)
        dt2srctf = [sourcetf[s][:,:]; zeros(size(sourcetf[s],2))]
##----------------
        for isr=1:size(sourcetf[s],2)
            # Each srctf has to be scaled with the velocity at
            #  same coordinates, etc.: vel^2*(dt^2/dh^2)
            dt2srctf[:,isr] .= fact[ijsrcs[s][isr,1],ijsrcs[s][isr,2]] .* dt2srctf[:,isr]
        end
##----------------

        # current sources for forward and adjoint calculations
        thishotijsrcs_fwd = ijsrcs[s]
        thishotijsrcs_adj = ijrecs[s]

        ntobs = size(obsrecv[s],1)
        ### ntobs+1 to mach the adjoint field time step
        thishotsrctf_adj = Array{Float64}(undef,ntobs+1,nrecs)
        thishotsrctf_adj[end,:] .= 0.0 ### ntobs+1 to mach the adjoint field time step

        ######################################
        ##      residuals calculation       ##
        ######################################
        #isnap = 0
        if verbose>0
            t1=time()
            println(" t1-t0: $(t1-t0)")
        end
        ## One more time step!
        ## nt+1 for the fin. diff. derivative w.r.t time in the adjoint !!
        @assert (size(dt2srctf,1)>=inpar.ntimesteps + 1  )

        for t=1:inpar.ntimesteps + 1 ## + 1 !!
            if verbose>0
                t%inpar.infoevery==0 && print("\rShot ",s," t: ",t," of ",inpar.ntimesteps)
            end
            
            ##==================================##
            ##           pressure
            ##==================================##

            ### arrays are swapped before being returned from oneiter_CPML!(),
            ###   that's why we need to return them (to make that happen)
            if useslowfd
                pold,pcur,pnew = oneiter_CPML!slow(nx,nz,fact,pnew,pold,pcur,dt2srctf,
                                                   #dpdx,dpdz,#d2pdx2,d2pdz2,
                                                   psi_x,psi_z,xi_x,xi_z,
                                                   cpml,thishotijsrcs_fwd,t)
            else
                pold,pcur,pnew = oneiter_CPML!(nx,nz,fact,pnew,pold,pcur,dt2srctf,
                                               #dpdx,dpdz,#d2pdx2,d2pdz2,
                                               psi_x,psi_z,xi_x,xi_z,
                                               cpml,thishotijsrcs_fwd,t)
            end
            
            ##========================
            ##### receivers
            ## skip the additional time step for receivers
            if t<=inpar.ntimesteps 
                for r=1:nrecs
                    ir = ijrecs[s][r,1]
                    jr = ijrecs[s][r,2]
                    receiv[s][t,r] = pcur[ir,jr]
                end            
            end
            
            ##========================
            ## save forward run
            ## t+1 because the first is zeros in the past, for adjoint
            pfwdsave[:,:,t+1] .= pcur

        end ##------- end time loop --------------

        if verbose>0
            t2=time()
        end

        ##============= ===========================
        ###------- Residuals -------------
        ## ddf = obsrecv - receiv
        ##residuals =  invC_d_onesrc * ddf
        for r=1:nrecs
            ##OLD: residuals[s][:,r] .= (receiv[s][:,r].-obsrecv[s][:,r]) ./ recstd.^2

            tmpdifcalobs .= receiv[s][:,r].-obsrecv[s][:,r]
            ## The followind line works but it allocates
            ##  residuals[s][:,r] .= invCovds[s] * tmpdifcalobs
            ## The next line produces zeros as output?!? Why??
            ##  mul!(residuals[s][:,r], invCovds[s], tmpdifcalobs)
            ## So using a second temporary array "tmpresid" to hold the
            ##    results and still avoid allocating...
            mul!(tmpresid, invCovds[s], tmpdifcalobs)
            ##residuals[s][:,r] .= tmpresid

            ## Source time function for adjoint
            ## REVERSE residuals in time
            ## last row of thishotsrctfresid must be already zero!
            #thishotsrctfresid[1:end-1,:] .= residuals[s][end:-1:1,r]
            thishotsrctf_adj[1:end-1,r] .= tmpresid[end:-1:1] 
        end    
        
        ##----------------
        for isr=1:size(thishotsrctf_adj,2)
            ## The adjoint source is *scaled only* by vel^2*dt^2 instead of vel^2*(dt^2/dh^2)
            thishotsrctf_adj[:,isr] .= thishotsrctf_adj[:,isr] .* vel2dt2[thishotijsrcs_adj[isr,1],thishotijsrcs_adj[isr,2]]
        end
        ##----------------

        if verbose>0
            t3=time()
            println("\n Forward calculation Time loop: ",t2-t1)
            println(" Residuals calculation Time loop: ",t3-t2)
            println(" Total residuals calculation Time loop: ",t3-t1)
        end

        ######################################
        ##      adjoint calculation         ##
        ######################################

        ## adjoint arrays
        adjold[:,:] .= 0.0
        adjcur[:,:] .= 0.0
        adjnew[:,:] .= 0.0

        # gradient for 1 shot
        curgrad .= 0.0

        ## PML arrays
        psi_x[:,:] .= 0.0
        psi_z[:,:] .= 0.0
        xi_x[:,:] .= 0.0
        xi_z[:,:] .= 0.0

        ##==================================##
        ## time loop
        if verbose>0
            t4=time()
        end
        nt = inpar.ntimesteps
        
        ## Adjoint actually going backward in time
        @assert (size(thishotsrctf_adj,1)>=nt-1 )
        for t = 1:nt
            if verbose>0
                t%inpar.infoevery==0 && print("\rShot ",s," t: ",t," of ",inpar.ntimesteps)
                t5 = time()
            end
                         
            ##==================================##
            ##           adjoint
            ##==================================##            
            if useslowfd
                adjold,adjcur,adjnew = oneiter_CPML!slow(nx,nz,fact,adjnew,adjold,
                                                         adjcur,thishotsrctf_adj,
                                                         #dpdx,dpdz,#d2pdx2,d2pdz2,
                                                         psi_x,psi_z,xi_x,xi_z,
                                                         cpml,thishotijsrcs_adj,t)
                
            else
                adjold,adjcur,adjnew = oneiter_CPML!(nx,nz,fact,adjnew,adjold,
                                                     adjcur,thishotsrctf_adj,
                                                     #dpdx,dpdz,#d2pdx2,d2pdz2,
                                                     psi_x,psi_z,xi_x,xi_z,
                                                     cpml,thishotijsrcs_adj,t)
            end
                        
            if verbose>1
                t6=time()
                println("\n Adjoint calculations: ",t6-t5)
            end

            ##==================================##
            ##          correlate
            ##==================================##
            ## p is shifted into future, so pcur is p at t+1
            ##dpcur2dt2[:,:] .=  (pcur .- 2.0.*pold .+ pveryold) ./ dt2
            for j=1:nz
                for i=1:nx
                    dpcur2dt2 = (pfwdsave[i,j,nt-t+1] - 2.0 * pfwdsave[i,j,nt-t+2] +
                        pfwdsave[i,j,nt-t+3]) / dt2
                    ## sum in time!
                    ## pointwise multiplication, integration in time...
                    curgrad[i,j] = curgrad[i,j] + (adjcur[i,j] * dpcur2dt2)
                end
            end 

            if verbose>1
                t7=time()
                println(" Correlating: ",t7-t6)
            end   

        end ##------- end time loop --------------

        ## tot gradient
        grad .= grad .+ curgrad

        if verbose>0
            t9=time()
            println(" Total adjoint solver time for 1 shot: ",t9-t4)
        end
        
    end ##------- for ishot=1:... --------------
    ##===========================================================##

    ## scale gradient
    grad .= (2.0 ./ vel3) .* grad

    if verbose>0
        t10 = time()
        println(" Init etc. : ",t0-tstart)
        
        println(" Total gradient calculation time for 1 shot: ",t10-tstart)
    end
        
    return grad ##,residuals
end


##########################################################################
##########################################################################
