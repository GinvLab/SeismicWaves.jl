
##==========================================================

verbose = 0
useslowfd = false

# type MyType
#     a::Int64
#     b::Float64
# end

# immutable, so no further changes after assignment
"""
  Parameters for acoustic wave simulations
"""
struct InpParamAcou
    ntimesteps::Int64
    nx::Int64
    nz::Int64
    dt::Float64
    dh::Float64
    savesnapshot::Bool
    snapevery::Int64
    boundcond::String
    freeboundtop::Bool
    infoevery::Int64
    #InpParam(ntimesteps=1,nx=1,ny=1,dx=1.0,dy=1.0) = new(ntimesteps,nx,ny,dx,dy)
end

struct BilinCoeff
    i::Array{Int64,1}
    j::Array{Int64,1}
    coe::Array{Float64,2}
    ##owner::Array{Int64,1}
end

struct CoefPML
    a_x::Array{Float64,1}
    b_x::Array{Float64,1}
    a_z::Array{Float64,1}
    b_z::Array{Float64,1}
    K_x::Array{Float64,1}
    K_z::Array{Float64,1}
    a_x_half::Array{Float64,1}
    b_x_half::Array{Float64,1}
    a_z_half::Array{Float64,1}
    b_z_half::Array{Float64,1}
    K_x_half::Array{Float64,1}
    K_z_half::Array{Float64,1}
    nptspml_x::Int64
    nptspml_z::Int64
    ipmlidxs::Array{Int64,1}
    jpmlidxs::Array{Int64,1}
end


struct GaussTaper
    xnptsgau::Int64
    ynptsgau::Int64
    leftdp::Array{Float64,1}
    rightdp::Array{Float64,1}
    bottomdp::Array{Float64,1}
end


##========================================

# function bilinearcoeff(hgrid::Float64, pts::Array{Float64,2})
#     ###idxsdarr::Array,pids::Array(Int64,1})
    
#     npt = size(pts,1)
#     iv = zeros(npt)
#     jv = zeros(npt)
#     coev = zeros(npt,4)

#     for p=1:npt
#         xreq=pts[p,1] # index starts from 1
#         yreq=pts[p,2]
#         xh=xreq/hgrid
#         yh=yreq/hgrid
#         i=floor(Int64,xh+1) # indices starts from 1
#         j=floor(Int64,yh+1) # indices starts from 1
#         #println("i,j $i $j   yh $yh   yreq $yreq   hgrid*j $(hgrid*j)")
#         xd=xh-(i-1) # indices starts from 1
#         yd=yh-(j-1) # indices starts from 1
        
#         iv[p] = i
#         jv[p] = j
#         coev[p,:] = [(1.0-xd)*(1.0-yd),  (1.0-yd)*xd,  (1.0-xd)*yd,  xd*yd]

#         # ## get the owner...
#         # npid = length(idxarr)
#         # for pr=1:npid
#         #     if (i in ra[pr][1]) & (j in ra[pr][2]) 
#         #         idxs[p] = pids[pr]
#         #     end
#         # end
#     end
    
#     bilin = BilinCoeff(iv,jv,coev)
    
#     # f[i,j]
#     # f[i+1,j]
#     # f[i,j+1]
#     # f[i+1,j+1]
#     return bilin
# end

##========================================

# function bilinear_interp(f::Union{DArray{Float64,2},Array{Float64,2}},
#                          hgrid::Float64, pt::Array{Float64,1})
#     xreq=pt[1] # index starts from 1
#     yreq=pt[2]
#     xh=xreq/hgrid
#     yh=yreq/hgrid
#     i=floor(Int64,xh+1) # indices starts from 1
#     j=floor(Int64,yh+1) # indices starts from 1
#     #println("i,j $i $j   yh $yh   yreq $yreq   hgrid*j $(hgrid*j)")
#     xd=xh-(i-1) # indices starts from 1
#     yd=yh-(j-1) # indices starts from 1
#     intval=f[i,j]*(1.0-xd)*(1.0-yd)+f[i+1,j]*(1.0-yd)*xd+f[i,j+1]*(1.0-xd)*yd+f[i+1,j+1]*xd*yd
#     #println("$xreq $yreq $xh $yh $i $j $xd $yd")
#     return intval
# end

##===============================================================================
##===============================================================================

"""
 Compute coefficients for Gaussian taper
"""
function initGausboundcon( ; decay::Float64=0.015)

    ## Damping region size in grid points
    xnptsgau = 21
    ynptsgau = 21

    xdist = collect(Float64,1:xnptsgau)
    ydist = collect(Float64,1:ynptsgau)

    xdamp = exp.( -((decay.*(xnptsgau .- xdist)).^2))
    ydamp = exp.( -((decay.*(ynptsgau .- ydist)).^2))

    leftdp   = copy(xdamp)
    rightdp  = xdamp[end:-1:1] 
    bottomdp = ydamp[end:-1:1] 
    
    gaubc = GaussTaper(xnptsgau,ynptsgau,leftdp,rightdp,bottomdp)
    return gaubc
end

##===============================================================================

"""
 Compute d, K and a parameters for CPML
"""
function calc_Kab_CPML(Ngrdpts::Integer,nptspml::Integer,gridspacing::Float64,dt::Float64,
                       Npower::Float64,d0::Float64,
                       alpha_max_pml::Float64,K_max_pml::Float64,onwhere::String )

    # # L = thickness of adsorbing layer
    # if onwhere=="grdpts"
    #     L = nptspml*gridspacing
    #     # distances 
    #     x = collect(range(0.0,step=gridspacing,length=nptspml))
    # elseif onwhere=="halfgrdpts"
    #     L = nptspml*gridspacing
    #     # distances 
    #     x = collect(range(gridspacing/2.0,step=gridspacing,length=nptspml))
    # end

    #Ngrdpts = 500
    #onwhere="halfgrd"


    K  = ones(Ngrdpts)
    b  = ones(Ngrdpts) #zeros(nx)
    a  = zeros(Ngrdpts)

    normdist = 0.0
    
    if onwhere=="halfgrd"
        tmpi = 0.5
    elseif onwhere=="ongrd"
        tmpi = 0.0
    else
        error("calc_Kab_CPML(): onwhere neither 'ongrd' nor 'halfgrd'")
    end

    for i=1:Ngrdpts
        ii = i + tmpi
        
        if ii<=Float64(nptspml)
            ## left or top border
            normdist = (nptspml-ii)/(nptspml-1)
            
        elseif ii>=Float64((Ngrdpts-nptspml+1))
            ## right or bottom border
            normdist = (ii-(Ngrdpts-nptspml+1))/(nptspml-1)

        else
            continue #normdist = 0.0 
        end
        
        d = d0 * normdist^Npower
        alpha =  alpha_max_pml * (1.0 - normdist) #.+ 0.1 .* alpha_max_pml ??
        
        K[i] = 1.0 + (K_max_pml - 1.0) * normdist^Npower
        b[i] = exp( - (d / K[i] + alpha) * dt )
        a[i] = d * (b[i]-1.0)/(K[i]*(d+K[i]*alpha))
        
    end

    return K,a,b
end
 
##====================================================================

"""
  Initialize the CPML absorbing boundaries
"""
function initCPML(inpar::InpParamAcou,vel_max::Float64,f0::Float64)
    
    dh = inpar.dh
    nx = inpar.nx
    nz = inpar.nz
    dt = inpar.dt

    if inpar.boundcond=="PML"

        ##############################
        #   Parameters for PML
        ##############################
        nptspml_x = 21 #convert(Int64,ceil((vel_max/f0)/dh))  
        nptspml_z = 21 #convert(Int64,ceil((vel_max/f0)/dh))  

        if verbose>0
            printbln(" Size of PML layers in grid points: $nptspml_x in x and $nptspml_z in z")
        end

        ##~~~~~~~~~~~~~~~~~
        Npower = 2.0 #2.0    
        K_max_pml = 1.0 # #1.0
        ## reflection coefficient (INRIA report section 6.1)
        ## http://hal.inria.fr/docs/00/07/32/19/PDF/RR-3471.pdf
        ## Collino 2001, refl. coeff for N PML layers (nodes...):
        # N = 5   R = 0.01
        # N = 10  R = 0.001
        # N = 20  R = 0.0001

        Rcoef = 0.0001  #0.001 for a PML thickness of 10 nodes
 
        @assert Npower>=1
        @assert K_max_pml>=1.0
        
        alpha_max_pml = 2.0*pi*(f0/2.0) #2.0*pi*(f0/2.0)

        # thickness of the PML layer in meters
        thickness_pml_x = (nptspml_x-1) * dh
        thickness_pml_z = (nptspml_z-1) * dh
        
        # compute d0 from INRIA report section 6.1 http://hal.inria.fr/docs/00/07/32/19/PDF/RR-3471.pdf
        d0_x = - (Npower + 1) * vel_max * log(Rcoef) / (2.0 * thickness_pml_x)
        d0_z = - (Npower + 1) * vel_max * log(Rcoef) / (2.0 * thickness_pml_z)

        ##############################
        #   Damping parameters
        ##############################    
        # --- damping in the x direction ---
        # assuming the number of grid points for PML is the same on 
        #    both sides    
        # damping profile at the grid points
        K_x,a_x,b_x = calc_Kab_CPML(nx,nptspml_x,dh,dt,Npower,d0_x,alpha_max_pml,K_max_pml,"ongrd")
        K_x_half,a_x_half,b_x_half = calc_Kab_CPML(nx,nptspml_x,dh,dt,Npower,d0_x,alpha_max_pml,K_max_pml,"halfgrd")
        
        # --- damping in the z direction ---
        # assuming the number of grid points for PML is the same on
        # both sides    
        # damping profile at the grid points
        K_z,a_z,b_z = calc_Kab_CPML(nz,nptspml_z,dh,dt,Npower,d0_z,alpha_max_pml,K_max_pml,"ongrd")
        K_z_half,a_z_half,b_z_half = calc_Kab_CPML(nz,nptspml_z,dh,dt,Npower,d0_z,alpha_max_pml,K_max_pml,"halfgrd")
        #### important \/ \/ \/
        if inpar.freeboundtop==true
            K_z[1:nptspml_z] .= 1.0
            a_z[1:nptspml_z] .= 0.0
            b_z[1:nptspml_z] .= 1.0
            K_z_half[1:nptspml_z] .= 1.0
            a_z_half[1:nptspml_z] .= 0.0
            b_z_half[1:nptspml_z] .= 1.0
        end

        ## struct of PML coefficients
        # ipmlidxs = append!(collect(2:nptspml_x), collect(nx-nptspml_x+1:nx-1))
        # jpmlidxs = append!(collect(2:nptspml_z), collect(nz-nptspml_z+1:nz-1))
        ipmlidxs = [2, nptspml_x, nx-nptspml_x+1, nx-1]
        jpmlidxs = [2, nptspml_z, nz-nptspml_z+1, nz-1]
        cpml = CoefPML(a_x,b_x,a_z,b_z,K_x,K_z, a_x_half,b_x_half,a_z_half,b_z_half,
                       K_x_half,K_z_half,nptspml_x,nptspml_z,ipmlidxs,jpmlidxs)
         
        ###########################

        # println(">>> WRITING PML COEFFICIENTS TO FILE!!! <<<")
        # writedlm("pml_coeff_X_julia.dat",[cpml.K_x cpml.a_x cpml.b_x cpml.K_x_half cpml.a_x_half cpml.b_x_half])
        # writedlm("pml_coeff_Z_julia.dat",[cpml.K_z cpml.a_z cpml.b_z cpml.K_z_half cpml.a_z_half cpml.b_z_half]) 

    else
        error("initCPML(): Expected inpar.boundcond==\"PML\" ")
    end

    return  nptspml_x, nptspml_z,cpml #K_x,K_z,a_x,a_z,b_x,b_z
end

##=======================================================================##

"""
  Solver for 2D acoustic wave equation (parameters: velocity only)
"""
function solveacoustic2D(inpar::InpParamAcou,ijsrcs::Array{Array{Int64,2},1},
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
        @show Cou
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
    dpdx = zeros(nx,nz)
    dpdz = zeros(nx,nz)
    d2pdx2 = zeros(nx,nz)
    d2pdz2 = zeros(nx,nz)
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

        ## ensure at least 10 pts per wavelengh  ????
        @assert dh <= vel_max/(10.0 * srcdomfreq[s])

        ##############################
        #   Parameters CPML
        ##############################
        if inpar.boundcond=="PML"
            
            f0 = srcdomfreq[s]
            nptspml_x, nptspml_z, cpml = initCPML(inpar,vel_max,f0)

            #@show cpml.ipmlidxs,cpml.jpmlidxs

            ##################################
            # PML arrays
            # Arrays with size of PML areas would be sufficient and save memory,
            #   however allocating arrays with same size than model simplifies
            #   the code in the loops
            dpdx[:,:] .= 0.0
            dpdz[:,:] .= 0.0
            d2pdx2[:,:] .= 0.0
            d2pdz2[:,:] .= 0.0
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
        ##    println(">>>>>>>>>ooo<<<<<<<<<<<")
        dt2srctf =  dt^2 .* sourcetf[s][:,:]  ## <<<===== ????? v^2 ????? =====##
        ## each srctf has to be scaled with the velocity at same coordinates
        for isr=1:size(sourcetf[s],2)
            dt2srctf[:,isr] = vel[ijsrcs[s][isr,1],ijsrcs[s][isr,2]].^2 .* dt2srctf[:,1]
        end
            
        #####################
        ##  Time loop
        #####################
        isnap = 0
        t1=time()
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
                                                       dpdx,dpdz,d2pdx2,d2pdz2,
                                                       psi_x,psi_z,xi_x,xi_z,
                                                       cpml,ijsrcs[s],t,inpar.freeboundtop)
                else
                    pold,pcur,pnew = oneiter_CPML!(nx,nz,fact,pnew,pold,pcur,dt2srctf,
                                                   dpdx,dpdz,d2pdx2,d2pdz2,
                                                   psi_x,psi_z,xi_x,xi_z,
                                                   cpml,ijsrcs[s],t,inpar.freeboundtop)  
                end
                
            elseif inpar.boundcond=="GauTap"
                oneiter_GAUSTAP!(nx,nz,fact,pnew,pold,pcur,dt2srctf,
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
        t2=time()
        if verbose>0
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
function gradadj_acoustic2D(inpar::InpParamAcou, obsrecv::Array{Array{Float64,2},1},recstd::Float64,
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

    if inpar.boundcond != "PML"
        error("gradadj_acoustic2D(): Boundary contitions must be PML for gradient computations.")
    end


    @assert length(sourcetf)==length(ijsrcs)
    @assert length(sourcetf)==length(ijrecs)
    @assert length(sourcetf)==length(srcdomfreq)

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
    totmem = nx*nz*nt*8/1000/1000/1000

    maxMEM = 8.0
    if totmem>maxMEM
        println("Requested mem for pfwdsave: ",totmem," GB, max. allowed ",maxMEM)
        error("Drea: Out of memory")
        return
    else
        if verbose>0
            println("Requested mem for pfwdsave: ",totmem," GB")
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
    
    # PML arrays
    # Arrays with size of PML areas would be sufficient and save memory,
    #  however allocating arrays with same size than model simplifies
    #  the code in the loops
    dpdx = zeros(nx,nz)
    dpdz = zeros(nx,nz)
    d2pdx2 = zeros(nx,nz)
    d2pdz2 = zeros(nx,nz)
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
    dpcur2dt2 = zeros(nx,nz)
    
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
        nptspml_x, nptspml_z, cpml = initCPML(inpar,vel_max,f0)
        
        ##################################
        # PML arrays
        # Arrays with size of PML areas would be sufficient and save memory,
        #   however allocating arrays with same size than model simplifies
        #   the code in the loops
        dpdx[:,:] .= 0.0
        dpdz[:,:] .= 0.0 
        d2pdx2[:,:] .= 0.0
        d2pdz2[:,:] .= 0.0
        psi_x[:,:] .= 0.0
        psi_z[:,:] .= 0.0
        xi_x[:,:] .= 0.0
        xi_z[:,:] .= 0.0

        ##################################
        # memory arrays
        pold[:,:] .= 0.0
        pcur[:,:] .= 0.0
        pnew[:,:] .= 0.0

        ## seismograms
        nrecs = size(ijrecs[s],1)
        receiv[s] = zeros(Float64,inpar.ntimesteps,nrecs)
        ## residuals for adjoint
        residuals[s] = zeros(inpar.ntimesteps,nrecs)
        
        ## Pre-scale source time function
        ##    ??? add 2 rows of zeros to match time step adjoint (0 and nt+1)
        dt2srctf = dt2 .* [sourcetf[s][:,:]; zeros(size(sourcetf[s],2))]
        ##dt2srctf =  dt2 .* sourcetf[s][:,:]  ## <<<===== ????? v^2 ????? =====##
        # Each srctf has to be scaled with the velocity at same coordinates
        for isr=1:size(sourcetf[s],2)
            dt2srctf[:,isr] = vel[ijsrcs[s][isr,1],ijsrcs[s][isr,2]].^2 .* dt2srctf[:,1]
        end      

        # current sources
        thishotijsrcs = ijsrcs[s]
        thishotijsrcs_adj = ijrecs[s]

        ntobs = size(obsrecv[s],1)
        thishotsrctfresid = Array{Float64}(undef,ntobs,nrecs)
        
        ######################################
        ##      residuals calculation       ##
        ######################################
        #isnap = 0
        t1=time() 
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
                                                    dpdx,dpdz,d2pdx2,d2pdz2,
                                                    psi_x,psi_z,xi_x,xi_z,
                                                    cpml,ijsrcs[s],t,inpar.freeboundtop)
            else
                pold,pcur,pnew = oneiter_CPML!(nx,nz,fact,pnew,pold,pcur,dt2srctf,
                                                dpdx,dpdz,d2pdx2,d2pdz2,
                                                psi_x,psi_z,xi_x,xi_z,
                                                cpml,ijsrcs[s],t,inpar.freeboundtop)
            end

            
            ##========================
            ##### receivers
            ## skip the additional time step for receivers
            if t<=inpar.ntimesteps 
                @inbounds for r=1:nrecs
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

        # residuals
        ## ddf = obsrecv - receiv
        ##residuals =  invC_d_onesrc * ddf
        @inbounds for r=1:nrecs
            residuals[s][:,r] .= (receiv[s][:,r].-obsrecv[s][:,r]) ./ recstd.^2
            ## Source time function for adjoint
            ##   (NO SCALING of srctf (i.e., dt2 .* srctf) for adjoint...? See eq. 18
            ##      Bunks et al., 1995 Geophysics, Multiscale seismic waveform inversion.
            ## FLIP residuals in time
            thishotsrctfresid[:,:] .= residuals[s][end:-1:1,r]
        end    
        ##    ??? add a row of zeros to match time step adjoint (nt+1)
        thishotsrctfresid = [thishotsrctfresid; zeros(1,size(thishotsrctfresid,2))]
        
        ## source time function for adjoint
        ##thishotsrctfresid = dt2 .* residuals[s][:,:]
        #tmpres = residuals[s][:,:]
        ##thishotsrctfresid = dt2 .* view(tmpres, size(tmpres,1):-1:1,:)
        #thishotsrctfresid = 1.0 .* view(tmpres, size(tmpres,1):-1:1,:)

        t2=time()
        if verbose>0
            println("\nresiduals calculation Time loop: ",t2-t1)
        end


        ######################################
        ##      adjoint calculation         ##
        ######################################
        ## pold[:,:] .= 0.0
        ## pcur[:,:] .= 0.0
        ## pnew[:,:] .= 0.0
        ## pveryold[:,:] .= 0.0
        # pveryold[:,:] .= pold

        ## adjoint arrays
        adjold[:,:] .= 0.0
        adjcur[:,:] .= 0.0
        adjnew[:,:] .= 0.0

        ## PML arrays
        dpdx[:,:] .= 0.0
        dpdz[:,:] .= 0.0
        d2pdx2[:,:] .= 0.0
        d2pdz2[:,:] .= 0.0
        psi_x[:,:] .= 0.0
        psi_z[:,:] .= 0.0
        xi_x[:,:] .= 0.0
        xi_z[:,:] .= 0.0

        ##==================================##
        ## time loop
        t1=time()
        nt = inpar.ntimesteps
        
        # ## compute pressure one step the future for dpcur2dt2
        # tpres = 1
        # oneiter_CPML!(nx,nz,fact,pnew,pold,pcur,


        # gradient for 1 shot
        curgrad .= 0.0

        ## Adjoint actually going backward in time
        @assert (size(thishotsrctfresid,1)>=nt-1 )
        for t = 1:nt
            if verbose>0
                t%inpar.infoevery==0 && print("\rShot ",s," t: ",t," of ",inpar.ntimesteps)
            end
            
            # ##==================================##
            # ##           pressure
            # ##==================================##
            # ## compute one step in the FUTURE for dpcur2dt2
            # pveryold[:,:] .= pold[:,:]
            # tpres = t+1
            # oneiter_CPML!( )
                        
            ##==================================##
            ##           adjoint
            ##==================================##            
            if useslowfd
                adjold,adjcur,adjnew = oneiter_CPML!slow(nx,nz,fact,adjnew,adjold,
                                                          adjcur,thishotsrctfresid,
                                                          dpdx,dpdz,d2pdx2,d2pdz2,
                                                          psi_x,psi_z,xi_x,xi_z,
                                                          cpml,thishotijsrcs_adj,
                                                          t,inpar.freeboundtop)
            else
                adjold,adjcur,adjnew = oneiter_CPML!(nx,nz,fact,adjnew,adjold,
                                                          adjcur,thishotsrctfresid,
                                                          dpdx,dpdz,d2pdx2,d2pdz2,
                                                          psi_x,psi_z,xi_x,xi_z,
                                                          cpml,thishotijsrcs_adj,
                                                          t,inpar.freeboundtop)
            end

            ##==================================##
            ##          correlate
            ##==================================##
            ## p is shifted into future, so pcur is p at t+1
            #dpcur2dt2[:,:] .=  (pcur .- 2.0.*pold .+ pveryold) ./ dt2
            @inbounds for j=1:nz
                @inbounds for i=1:nx
                    # dpcur2dt2[i,j] = (pfwdsave[i,j,nt-t] - 2.0 * pfwdsave[i,j,nt-t+1] +
                    #                    pfwdsave[i,j,nt-t+2]) / dt2
                    dpcur2dt2[i,j] = (pfwdsave[i,j,nt-t+1] - 2.0 * pfwdsave[i,j,nt-t+2] +
                                      pfwdsave[i,j,nt-t+3]) / dt2
                    ## sum in time!
                    ## pointwise multiplication, integration in time...
                    curgrad[i,j] = curgrad[i,j] + (adjcur[i,j] * dpcur2dt2[i,j])
                end
            end

            # if (t%200==0) & s==1
            #     println("\n\n REMOVE using PyPlot at the top of the file! \n\n")
            #     figure(figsize=(12,9))
            #     subplot(221)
            #     title(string("pressure  time ", nt-t))
            #     gvmax = maximum(abs.(pfwdsave[:,:,nt-t]))
            #     imshow(permutedims(pfwdsave[:,:,nt-t]),vmin=-gvmax,vmax=gvmax,cmap=get_cmap("RdBu"))
            #     colorbar()
            #     subplot(222)
            #     title(string("adjoint time ",t))
            #     gvmax = maximum(abs.(adjcur))
            #     imshow(permutedims(adjcur),vmin=-gvmax,vmax=gvmax,cmap=get_cmap("RdBu")) 
            #     colorbar()
            #     subplot(223)
            #     title("(adjcur .* dpcur2dt2)") #dpcur2dt2")
            #     gvmax = maximum(abs.(adjcur .* dpcur2dt2))
            #     imshow(permutedims(adjcur .* dpcur2dt2),vmin=-gvmax,vmax=gvmax,cmap=get_cmap("RdBu")) 
            #     colorbar()
            #     subplot(224)
            #     title("gradient")
            #     gvmax = maximum(abs.(curgrad))
            #     imshow(permutedims(curgrad),vmin=-gvmax,vmax=gvmax,cmap=get_cmap("RdBu")) 
            #     colorbar()
            #     #show(block=true)
            # end

        end ##------- end time loop --------------

        ## scale gradient
        grad .= grad .+ 2.0 ./ vel.^3 .* curgrad
        

        t2=time()
        if verbose>0
            println("\nAdjoint calculation time loop: ",t2-t1)
        end

        
    end ##------- for ishot=1:... --------------
    ##===========================================================##

    ## scale gradient
    #grad = - 2.0 ./ vel.^3 .* curgrad
        
    return grad,residuals
end


##########################################################################
##########################################################################

"""
  Calculate one iteration - one time step - for the acoustic wave 
   equation with reflective boundaries
"""
function oneiter_reflbound!(nx::Int64,nz::Int64,fact::Array{Float64,2},pnew::Array{Float64,2},
                            pold::Array{Float64,2},pcur::Array{Float64,2},dt2srctf::Array{Float64,2},
                            ijsrcs::Array{Int64,2},t::Int64,freeboundtop::Bool)
    
    if freeboundtop==true
        ##----------------------------------
        ## free surface boundary cond.
        j=1
        pcur[:,j] .= 0.0
        @inbounds for i = 2:nx-1
            dpdx = pcur[i+1,j]-2.0*pcur[i,j]+pcur[i-1,j]
            dpdz = pcur[i,j+1]-2.0*pcur[i,j]+ 0.0 #pcur[i,j-1]
            # update pressure
            pnew[i,j] = 2.0*pcur[i,j] -pold[i,j] +
                fact[i,j]*(dpdx) + fact[i,j]*(dpdz)
        end
    end  ##----------------------------------

    ## space loop excluding boundaries
    @inbounds for j = 2:nz-1
        @inbounds for i = 2:nx-1
            ## second derivative stencil
            d2pdx2u = pcur[i+1,j]-2.0*pcur[i,j]+pcur[i-1,j]
            d2pdz2u = pcur[i,j+1]-2.0*pcur[i,j]+pcur[i,j-1]

            # update pressure
            pnew[i,j] = 2.0*pcur[i,j] -pold[i,j] +
                fact[i,j]*(d2pdx2u) + fact[i,j]*(d2pdz2u) 
        end
    end

    # inject source(s)
    @inbounds for l=1:size(dt2srctf,2)
        isrc = ijsrcs[l,1]
        jsrc = ijsrcs[l,2]
        pnew[isrc,jsrc] = pnew[isrc,jsrc] + dt2srctf[t,l]       
    end

    # assign the new pold and pcur
    @inbounds pold .= pcur
    @inbounds pcur .= pnew
    return
end

##########################################################################
##########################################################################

"""
  Calculate one iteration - one time step - for the acoustic wave 
   equation with Gaussian taper boundary condition
"""
function oneiter_GAUSSTAP!(nx::Int64,nz::Int64,fact::Array{Float64,2},pnew::Array{Float64,2},
                            pold::Array{Float64,2},pcur::Array{Float64,2},dt2srctf::Array{Float64,2},
                            ijsrcs::Array{Int64,2},t::Int64,gaubc::GaussTaper,freeboundtop::Bool)
    

    if freeboundtop==true
        ##----------------------------------
        ## free surface boundary cond.
        j=1
        pcur[:,j] .= 0.0
        @inbounds for i = 2:nx-1
            dpdx = pcur[i+1,j]-2.0*pcur[i,j]+pcur[i-1,j]
            dpdz = pcur[i,j+1]-2.0*pcur[i,j]+ 0.0 #pcur[i,j-1]
            # update pressure
            pnew[i,j] = 2.0*pcur[i,j] -pold[i,j] +
                fact[i,j]*(dpdx) + fact[i,j]*(dpdz)
        end
    end  ##----------------------------------

    ## space loop excluding boundaries
    @inbounds for j = 2:nz-1
        @inbounds for i = 2:nx-1
            ## second derivative stencil
            d2pdx2u = pcur[i+1,j]-2.0*pcur[i,j]+pcur[i-1,j]
            d2pdz2u = pcur[i,j+1]-2.0*pcur[i,j]+pcur[i,j-1]

            # update pressure
            pnew[i,j] = 2.0*pcur[i,j] -pold[i,j] +
                fact[i,j]*(d2pdx2u) + fact[i,j]*(d2pdz2u) 
        end
    end

    ## Apply Gaussian taper damping as boundary condition
    pnew[1:gaubc.xnptsgau,:]          .*= gaubc.leftdp 
    pnew[end-gaubc.xnptsgau+1:end,:]  .*= gaubc.rightdp 
    pnew[:,end-gaubc.ynptsgau+1:end]  .*= gaubc.bottomdp
    
    
    # inject source(s)
    @inbounds for l=1:size(dt2srctf,2)
        isrc = ijsrcs[l,1]
        jsrc = ijsrcs[l,2]
        pnew[isrc,jsrc] = pnew[isrc,jsrc] + dt2srctf[t,l]       
    end

    # assign the new pold and pcur
    @inbounds pold .= pcur
    @inbounds pcur .= pnew
    return
end

##########################################################################
##########################################################################


"""
  Calculate one iteration - one time step - for the acoustic wave 
   equation with CPML absorbing boundary conditions
"""
function oneiter_CPML!(nx::Int64,nz::Int64,fact::Array{Float64,2},pnew::Array{Float64,2},
                        pold::Array{Float64,2},pcur::Array{Float64,2},dt2srctf::Array{Float64,2},
                        dpdx::Array{Float64,2},dpdz::Array{Float64,2},
                        d2pdx2::Array{Float64,2},d2pdz2::Array{Float64,2},
                        psi_x::Array{Float64,2},psi_z::Array{Float64,2},
                        xi_x::Array{Float64,2},xi_z::Array{Float64,2},
                        cpml::CoefPML,
                        ijsrcs::Array{Int64,2},t::Int64,freeboundtop::Bool)
                       

    #
    # see seismic_CPML Fortran codes
    #
    
    # Komatitsch, D., and Martin, R., 2007, An unsplit convolutional perfectly
    # matched layer improved at grazing incidence for the seismic wave
    # equation: Geophysics, 72, SM155–SM167. doi:10.1190/1.2757586
    #
       
    # Pasalic, D., and McGarry, R., 2010, Convolutional perfectly matched layer
    # for isotropic and anisotropic acoustic wave equations: 80th Annual
    # International Meeting, SEG, Expanded Abstracts, 2925–2929.


    # if freeboundtop==true
    #     ##----------------------------------
    #     ## free surface boundary cond.
    #     j=1
    #     #pcur[:,j] .= 0.0  ## ??
    #     @inbounds for i = 2:nx-1
    #         dpdx[i,j] = pcur[i+1,j]-2.0*pcur[i,j]+pcur[i-1,j] ## ==0.0??
    #         dpdz[i,j] = pcur[i,j+1]-2.0*pcur[i,j]+ 0.0 #pcur[i,j-1]
    #         # update pressure
    #         pnew[i,j] = 2.0*pcur[i,j] -pold[i,j] + fact[i,j]*(dpdx[i,j] + dpdz[i,j])
    #         #@show i,pnew[i,j],dpdx[i,j],dpdz[i,j]
    #     end
    # end  ##----------------------------------
    
    ##====================================================
    # pmlranges[1,1] = 2:nptspml_x         # exclude boundary
    # pmlranges[2,1] = nz-nptspml_x+1:nx-1 # exclude boundary
    # pmlranges[1,2] = 2:nptspml_z         # exclude boundary
    # pmlranges[2,2] = nz-nptspml_z+1:nz-1 # exclude boundary
   
    ##====================================================
    
    # ## space loop excluding boundaries
    # @inbounds  for j = 2:nz-1 # 3:nz-2
    #      @inbounds for i = 2:nx-1 # 3:nx-2

             #(-f[i-2]+16*f[i-1]-30*f[i]+16*f[i+1]-1*f[i+2])/(12*1.0*h**2)

             ## 4th order
             # d2pdx2 = ( -pcur[i-2,j]+16*pcur[i-1,j]-30*pcur[i,j]+16*pcur[i+1,j]-pcur[i+2,j] )/12
             # d2pdz2 = ( -pcur[i,j-2]+16*pcur[i,j-1]-30*pcur[i,j]+16*pcur[i,j+1]-pcur[i,j+2] )/12
             # dpsidx = (psi_x[i-2,j]-8*psi_x[i-1,j]+8*psi_x[i+1,j]-psi_x[i+2,j])/12
             # dpsidz = (psi_z[i,j-2]-8*psi_z[i,j-1]+8*psi_z[i,j+1]-psi_z[i,j+2])/12
            
             # d2pdx2 = pcur[i+1,j]-2.0*pcur[i,j]+pcur[i-1,j]            
             # d2pdz2 = pcur[i,j+1]-2.0*pcur[i,j]+pcur[i,j-1]
             # dpsidx = (psi_x[i+1,j] - psi_x[i-1,j])*0.5
             # dpsidz = (psi_z[i,j+1] - psi_z[i,j-1])*0.5



    ##====================================================
    ## Loop only withing PML layers to spare calculating zeros...
    # ipmlidxs = [2, nptspml_x, nx-nptspml_x+1, nx-1]
    # jpmlidxs = [2, nptspml_z, nz-nptspml_z+1, nz-1]
    
    ## X
    @inbounds for ii=(1,3)
        @inbounds  for j = 1:nz # 1:nz !!
            @inbounds for i = cpml.ipmlidxs[ii]:cpml.ipmlidxs[ii+1]
                dpdx[i,j] = pcur[i+1,j]-pcur[i,j] 
                psi_x[i,j] = cpml.b_x_half[i] / cpml.K_x_half[i] * psi_x[i,j] + cpml.a_x_half[i] * dpdx[i,j]
            end
        end
    end

    ## Z
    @inbounds for jj=(1,3)
        @inbounds for j = cpml.jpmlidxs[jj]:cpml.jpmlidxs[jj+1]
            @inbounds  for i = 1:nx # 1:nx !!
                dpdz[i,j] = pcur[i,j+1]-pcur[i,j]  
                psi_z[i,j] = cpml.b_z_half[j] / cpml.K_z_half[j] * psi_z[i,j] + cpml.a_z_half[j] * dpdz[i,j]
            end
        end
    end
  
    ##====================================================
    ## Calculate PML stuff only on the borders...
    ## X borders
    @inbounds for ii=(1,3)
        @inbounds  for j = 2:nz-1 # 2:nz-1 !!
            @inbounds for i = cpml.ipmlidxs[ii]:cpml.ipmlidxs[ii+1]
                
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
    end

    ## Calculate PML stuff only on the borders...
    ## Z borders
    @inbounds for jj=(1,3)
        @inbounds for j = cpml.jpmlidxs[jj]:cpml.jpmlidxs[jj+1]
            #@inbounds  for i = 2:nx-1 # 2:nx-1 !!
            ##--------------------------------------------------------------------------
            ## EXCLUDE CORNERS, because already visited in the previous X-borders loop!
            ##  (It would lead to wrong accumulation of pnew[i,j], etc. otherwise...)
            ##--------------------------------------------------------------------------
            @inbounds  for i = cpml.ipmlidxs[2]+1:cpml.ipmlidxs[4]-1
                
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
    end

    ## Calculate internal part
    @inbounds for j = cpml.jpmlidxs[2]+1:cpml.jpmlidxs[3]-1    #2:nz-1 
        @inbounds for i = cpml.ipmlidxs[2]+1:cpml.ipmlidxs[3]-1   #2:nx-1 

            d2pdx2 = pcur[i+1,j]-2.0*pcur[i,j]+pcur[i-1,j]            
            d2pdz2 = pcur[i,j+1]-2.0*pcur[i,j]+pcur[i,j-1]
            
            pnew[i,j] = 2.0*pcur[i,j] -pold[i,j] +
                fact[i,j]*(d2pdx2 + d2pdz2)
        end
    end

    
    ##====================================================
    
    # inject source(s)
    @inbounds for l=1:size(dt2srctf,2)
        isrc = ijsrcs[l,1]
        jsrc = ijsrcs[l,2]
        pnew[isrc,jsrc] = pnew[isrc,jsrc] + dt2srctf[t,l] 
    end

    # assign the new pold and pcur
    # po = pointer_from_objref(pold)
    # pc = pointer_from_objref(pcur)
    # pn = pointer_from_objref(pnew)    
    # pold = unsafe_pointer_to_objref(pc) ## pointers exchange???
    # pcur = unsafe_pointer_to_objref(pn)
    # pnew = unsafe_pointer_to_objref(po)
    
    # assign the new pold and pcur
    ## https://discourse.julialang.org/t/swap-array-contents/7774/7
    ## kind of swapping array pointers, so NEED TO RETURN NEW BINDINGS,
    ##  otherwise the exchange is lost!
    pold,pcur,pnew = pcur,pnew,pold

    ### this is slower
    # @inbounds pold .= pcur 
    # @inbounds pcur .= pnew

    return pold,pcur,pnew
end


################################################################################################
################################################################################################
################################################################################################


function oneiter_CPML!slow(nx::Int64,nz::Int64,fact::Array{Float64,2},pnew::Array{Float64,2},
                        pold::Array{Float64,2},pcur::Array{Float64,2},dt2srctf::Array{Float64,2},
                        dpdx::Array{Float64,2},dpdz::Array{Float64,2},
                        d2pdx2::Array{Float64,2},d2pdz2::Array{Float64,2},
                        psi_x::Array{Float64,2},psi_z::Array{Float64,2},
                        xi_x::Array{Float64,2},xi_z::Array{Float64,2},
                        cpml::CoefPML,
                        ijsrcs::Array{Int64,2},t::Int64,freeboundtop::Bool)


    ##-------------------------------------------
    ###############################
    ### SLOW but simple VERSION
    ###############################

    ## compute current psi_x and psi_z first (need derivatives next)
    @inbounds  for j = 2:nz-1 # 3:nz-2
        @inbounds for i = 2:nx-1 # 3:nx-2
            dpdx[i,j] = pcur[i+1,j]-pcur[i,j] 
            dpdz[i,j] = pcur[i,j+1]-pcur[i,j]
            psi_x[i,j] = cpml.b_x_half[i] / cpml.K_x_half[i] * psi_x[i,j] + cpml.a_x_half[i] * dpdx[i,j]
            psi_z[i,j] = cpml.b_z_half[j] / cpml.K_z_half[j] * psi_z[i,j] + cpml.a_z_half[j] * dpdz[i,j]
        end
    end

    @inbounds  for j = 2:nz-1 # 3:nz-2
        @inbounds for i = 2:nx-1 # 3:nx-2

            d2pdx2 = pcur[i+1,j]-2.0*pcur[i,j]+pcur[i-1,j]            
            d2pdz2 = pcur[i,j+1]-2.0*pcur[i,j]+pcur[i,j-1]

            dpsidx = psi_x[i,j] - psi_x[i-1,j] 
            dpsidz = psi_z[i,j] - psi_z[i,j-1]

            xi_x[i,j] = cpml.b_x[i] / cpml.K_x_half[i] * xi_x[i,j] + cpml.a_x[i] * (d2pdx2 + dpsidx)
            xi_z[i,j] = cpml.b_z[j] / cpml.K_z_half[i] * xi_z[i,j] + cpml.a_z[j] * (d2pdz2 + dpsidz)
            
            damp = fact[i,j] * (dpsidx + dpsidz + xi_x[i,j] + xi_z[i,j])

            # update pressure
            pnew[i,j] = 2.0*pcur[i,j] -pold[i,j] +
                fact[i,j]*(d2pdx2 + d2pdz2) + damp
            
        end
    end

    ###############################
    ##====================================================
    
    # inject source(s)
    @inbounds for l=1:size(dt2srctf,2)
        isrc = ijsrcs[l,1]
        jsrc = ijsrcs[l,2]
        pnew[isrc,jsrc] = pnew[isrc,jsrc] + dt2srctf[t,l] 
    end
    
    # assign the new pold and pcur
    ## https://discourse.julialang.org/t/swap-array-contents/7774/7
    ## kind of swapping array pointers, so NEED TO RETURN NEW BINDINGS,
    ##  otherwise the exchange is lost!
    pold,pcur,pnew = pcur,pnew,pold

    return pold,pcur,pnew
end

##======================================

