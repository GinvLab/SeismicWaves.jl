
###################################################################

const verbose = 0
const useslowfd = false
if useslowfd
    println("\n useslowfd = $useslowfd \n")
end

###################################################################
"""
  Parameters for acoustic wave simulations
"""
Base.@kwdef struct InpParamAcou
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
    smoothgrad::Bool
    #InpParam(ntimesteps=1,nx=1,ny=1,dx=1.0,dy=1.0) = new(ntimesteps,nx,ny,dx,dy)
end

###################################################################

struct BilinCoeff
    i::Array{Int64,1}
    j::Array{Int64,1}
    coe::Array{Float64,2}
    ##owner::Array{Int64,1}
end

###################################################################

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

###################################################################

struct GaussTaper
    xnptsgau::Int64
    ynptsgau::Int64
    leftdp::Array{Float64,1}
    rightdp::Array{Float64,1}
    bottomdp::Array{Float64,2}  ## 2 because it's a row vector
end

###################################################################

function acoumisfitfunc(inpar::InpParamAcou,ijsrcs::Array{Array{Int64,2},1},
                        vel::Array{Float64,2}, ijrecs::Array{Array{Int64,2},1},
                        sourcetf::Array{Array{Float64,2},1}, srcdomfreq::Array{Float64,1},
                        obsrecv::Array{Array{Float64,2},1},
                        invCovds::Union{Vector{Matrix{Float64}},Vector{Diagonal{Float64}}};
                        runparallel::Bool=false)

    # compute synthetic data
    seismrecv = solveacoustic2D(inpar, ijsrcs, vel, ijrecs, sourcetf,srcdomfreq,
                                runparallel=runparallel)
    
    # compute misfit
    misf = 0.0
    nshots = length(ijsrcs) # length, not size because array of arrays
    nptsseismogram = size(obsrecv[1],1)
    tmp1 = zeros(nptsseismogram)
    difcalobs = zeros(nptsseismogram)
    for s=1:nshots
        nrec = size(ijrecs[s],1) # nrec for each shot
        for r=1:nrec
            difcalobs .= seismrecv[s][:,r].-obsrecv[s][:,r]
            mul!(tmp1, invCovds[s], difcalobs)
            misf += dot(difcalobs,tmp1)
        end
    end
    misf = 0.5 * misf

    return  misf
end

##===============================================================================

"""
  Solver for 2D acoustic wave equation (parameters: velocity only). 
"""
function solveacoustic2D(inpar::InpParamAcou,ijsrcs::Array{Array{Int64,2},1},
                         vel::Array{Float64,2}, ijrecs::Array{Array{Int64,2},1},
                         sourcetf::Array{Array{Float64,2},1}, srcdomfreq::Array{Float64,1};
                         runparallel::Bool=false)

    if runparallel
        output = solveacoustic2D_parallel(inpar,ijsrcs,vel,ijrecs,sourcetf,srcdomfreq)

        if inpar.savesnapshot==true
            receiv,psave = output[1],output[2]
            return receiv,psave
        end

    else
        output = solveacoustic2D_serial(inpar,ijsrcs,vel,ijrecs,sourcetf,srcdomfreq)

        if inpar.savesnapshot==true
            receiv,psave = output[1],output[2]
            return receiv,psave
        end
    end

    return output
end

##===============================================================================

"""
  Solver for computing the gradient of the misfit function for the acoustic 
   wave equation using the adjoint state method.
"""
function gradacoustic2D(inpar::InpParamAcou, obsrecv::Array{Array{Float64,2},1},
                        invCovds::Union{Vector{Matrix{Float64}},Vector{Diagonal{Float64}}}, 
                        ijsrcs::Array{Array{Int64,2},1},
                        vel::Array{Float64,2}, ijrecs::Array{Array{Int64,2},1},
                        sourcetf::Array{Array{Float64,2},1}, srcdomfreq::Array{Float64,1};
                        runparallel::Bool=false, 
                        calcpenergy::Bool=false)

    if runparallel
        if calcpenergy
            grad,penergy = gradacoustic2D_parallel(inpar,obsrecv,invCovds,ijsrcs,vel,ijrecs,sourcetf,srcdomfreq,
                                                   calcpenergy=true)
            return grad,penergy
        else
            grad = gradacoustic2D_parallel(inpar,obsrecv,invCovds,ijsrcs,vel,ijrecs,sourcetf,srcdomfreq)
        end
    else
        grad = gradacoustic2D_serial(inpar,obsrecv,invCovds,ijsrcs,vel,ijrecs,sourcetf,srcdomfreq)
    end

    return grad
end

########################################################################

"""
 Compute coefficients for Gaussian taper
"""
function initGausboundcon(; npad=21 ) #; decay::Float64=0.015)  # 0.015

    ## keep the lowest value to 0.85
    ## 0.85 = exp( -(decay*npad)^2 )
    decay = 0.42/npad

    ## Damping region size in grid points
    xnptsgau = npad
    ynptsgau = npad

    xdist = collect(Float64,1:xnptsgau)
    ydist = collect(Float64,1:ynptsgau)

    xdamp = exp.( -( (decay.*(xnptsgau .- xdist)).^2))
    ydamp = exp.( -( (decay.*(ynptsgau .- ydist)).^2))

    leftdp   = copy(xdamp)
    rightdp  = xdamp[end:-1:1] 
    bottomdp = reshape(ydamp[end:-1:1],1,:)
    
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
function initCPML(inpar::InpParamAcou,vel_max::Union{Float64,Nothing},f0::Union{Float64,Nothing})
    
    dh = inpar.dh
    nx = inpar.nx
    nz = inpar.nz
    dt = inpar.dt

    @assert inpar.boundcond=="PML"

    ##############################
    #   Parameters for PML
    ##############################
    nptspml_x = 21 # N = 20  R = 0.0001  #convert(Int64,ceil((vel_max/f0)/dh))  
    nptspml_z = 21 # N = 20  R = 0.0001  #convert(Int64,ceil((vel_max/f0)/dh))  

    # ipmlidxs = append!(collect(2:nptspml_x), collect(nx-nptspml_x+1:nx-1))
    # jpmlidxs = append!(collect(2:nptspml_z), collect(nz-nptspml_z+1:nz-1))
    ipmlidxs = [2, nptspml_x, nx-nptspml_x+1, nx-1]
    jpmlidxs = [2, nptspml_z, nz-nptspml_z+1, nz-1]
    
    if vel_max==nothing && f0==nothing
        return ipmlidxs,jpmlidxs
    end

    if verbose>0
        println(" Size of PML layers in grid points: $nptspml_x in x and $nptspml_z in z")
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
    cpml = CoefPML(a_x,b_x,a_z,b_z,K_x,K_z, a_x_half,b_x_half,a_z_half,b_z_half,
                   K_x_half,K_z_half,nptspml_x,nptspml_z,ipmlidxs,jpmlidxs)
    
    return cpml 
end

##########################################################################
##########################################################################

"""
  Calculate one iteration - one time step - for the acoustic wave 
   equation with reflective boundaries
"""
function oneiter_reflbound!(nx::Int64,nz::Int64,fact::Array{Float64,2},pnew::Array{Float64,2},
                            pold::Array{Float64,2},pcur::Array{Float64,2},dt2srctf::Array{Float64,2},
                            ijsrcs::Array{Int64,2},t::Int64) #,freeboundtop::Bool)
    
    # if freeboundtop==true
    #     ##----------------------------------
    #     ## free surface boundary cond.
    #     j=1
    #     pcur[:,j] .= 0.0
    #      for i = 2:nx-1
    #         dpdx = pcur[i+1,j]-2.0*pcur[i,j]+pcur[i-1,j]
    #         dpdz = pcur[i,j+1]-2.0*pcur[i,j]+ 0.0 #pcur[i,j-1]
    #         # update pressure
    #         pnew[i,j] = 2.0*pcur[i,j] -pold[i,j] +
    #             fact[i,j]*(dpdx) + fact[i,j]*(dpdz)
    #     end
    # end  ##----------------------------------

    ## space loop excluding boundaries
     for j = 2:nz-1
         for i = 2:nx-1
            ## second derivative stencil
            d2pdx2 = pcur[i+1,j]-2.0*pcur[i,j]+pcur[i-1,j]
            d2pdz2 = pcur[i,j+1]-2.0*pcur[i,j]+pcur[i,j-1]

            # update pressure
            pnew[i,j] = 2.0*pcur[i,j] -pold[i,j] +
                fact[i,j]*(d2pdx2) + fact[i,j]*(d2pdz2) 
        end
    end

    # inject source(s)
    for l=1:size(dt2srctf,2)
        isrc = ijsrcs[l,1]
        jsrc = ijsrcs[l,2]
        pnew[isrc,jsrc] = pnew[isrc,jsrc] + dt2srctf[t,l]
    end

    # assign the new pold and pcur
    pold .= pcur
    pcur .= pnew
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
                            ijsrcs::Array{Int64,2},t::Int64,gaubc::GaussTaper) #,freeboundtop::Bool)
    
    # if freeboundtop==true
    #     ##----------------------------------
    #     ## free surface boundary cond.
    #     j=1
    #     pcur[:,j] .= 0.0
    #      for i = 2:nx-1
    #         dpdx = pcur[i+1,j]-2.0*pcur[i,j]+pcur[i-1,j]
    #         dpdz = pcur[i,j+1]-2.0*pcur[i,j]+ 0.0 #pcur[i,j-1]
    #         # update pressure
    #         pnew[i,j] = 2.0*pcur[i,j] -pold[i,j] +
    #             fact[i,j]*(dpdx) + fact[i,j]*(dpdz)
    #     end
    # end  ##----------------------------------


    ## space loop excluding boundaries
     for j = 2:nz-1
         for i = 2:nx-1
            ## second derivative stencil
            d2pdx2 = pcur[i+1,j]-2.0*pcur[i,j]+pcur[i-1,j]
            d2pdz2 = pcur[i,j+1]-2.0*pcur[i,j]+pcur[i,j-1]

            # update pressure
            pnew[i,j] = 2.0*pcur[i,j] -pold[i,j] +
                fact[i,j]*(d2pdx2 + d2pdz2) 
        end
    end

    ## Apply Gaussian taper damping as boundary condition
    pnew[1:gaubc.xnptsgau,:]         .*= gaubc.leftdp 
    pnew[end-gaubc.xnptsgau+1:end,:] .*= gaubc.rightdp
    pnew[:,end-gaubc.ynptsgau+1:end] .*= gaubc.bottomdp
    
    pold[1:gaubc.xnptsgau,:]          .*= gaubc.leftdp 
    pold[end-gaubc.xnptsgau+1:end,:]  .*= gaubc.rightdp
    pold[:,end-gaubc.ynptsgau+1:end]  .*= gaubc.bottomdp

    pcur[1:gaubc.xnptsgau,:]          .*= gaubc.leftdp 
    pcur[end-gaubc.xnptsgau+1:end,:]  .*= gaubc.rightdp
    pcur[:,end-gaubc.ynptsgau+1:end]  .*= gaubc.bottomdp
    
    # inject source(s)
    for l=1:size(dt2srctf,2)
        isrc = ijsrcs[l,1]
        jsrc = ijsrcs[l,2]
        pnew[isrc,jsrc] = pnew[isrc,jsrc] + dt2srctf[t,l]
    end
 
    ## kind of swapping array pointers, so NEED TO RETURN NEW BINDINGS,
    ##  otherwise the exchange is lost!
    pold,pcur,pnew = pcur,pnew,pold
    
    return pold,pcur,pnew
end

##########################################################################
##########################################################################


"""
  Calculate one iteration - one time step - for the acoustic wave 
   equation with CPML absorbing boundary conditions
"""
function oneiter_CPML!(nx::Int64,nz::Int64,fact::Array{Float64,2},pnew::Array{Float64,2},
                       pold::Array{Float64,2},pcur::Array{Float64,2},dt2srctf::Array{Float64,2},
                       psi_x::Array{Float64,2},psi_z::Array{Float64,2},
                       xi_x::Array{Float64,2},xi_z::Array{Float64,2},
                       cpml::CoefPML,ijsrcs::Array{Int64,2},t::Int64)
    

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
    #      for i = 2:nx-1
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
    #   for j = 2:nz-1 # 3:nz-2
    #       for i = 2:nx-1 # 3:nx-2

            ###(-f[i-2]+16*f[i-1]-30*f[i]+16*f[i+1]-1*f[i+2])/(12*1.0*h**2)

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
     for ii=(1,3)
          for j = 1:nz # 1:nz !!
               for i = cpml.ipmlidxs[ii]:cpml.ipmlidxs[ii+1]
                dpdx = pcur[i+1,j]-pcur[i,j] 
                psi_x[i,j] = cpml.b_x_half[i] / cpml.K_x_half[i] * psi_x[i,j] + cpml.a_x_half[i] * dpdx
            end
        end
    end

    ## Z
     for jj=(1,3)
         for j = cpml.jpmlidxs[jj]:cpml.jpmlidxs[jj+1]
               for i = 1:nx # 1:nx !!
                dpdz = pcur[i,j+1]-pcur[i,j]  
                psi_z[i,j] = cpml.b_z_half[j] / cpml.K_z_half[j] * psi_z[i,j] + cpml.a_z_half[j] * dpdz
            end
        end
    end
  
    ##====================================================
    ## Calculate PML stuff only on the borders...
    ## X borders
     for ii=(1,3)
          for j = 2:nz-1 # 2:nz-1 !!
               for i = cpml.ipmlidxs[ii]:cpml.ipmlidxs[ii+1]
                
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
    end

    ## Calculate PML stuff only on the borders...
    ## Z borders
     for jj=(1,3)
         for j = cpml.jpmlidxs[jj]:cpml.jpmlidxs[jj+1]
            #  for i = 2:nx-1 # 2:nx-1 !!
            ##--------------------------------------------------------------------------
            ## EXCLUDE CORNERS, because already visited in the previous X-borders loop!
            ##  (It would lead to wrong accumulation of pnew[i,j], etc. otherwise...)
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
    end

    ##----------------------------------------------------------
    ## Calculate stuff in the internal part of the model
     for j = cpml.jpmlidxs[2]+1:cpml.jpmlidxs[3]-1    #2:nz-1 
          for i = cpml.ipmlidxs[2]+1:cpml.ipmlidxs[3]-1   #2:nx-1 

            d2pdx2 = pcur[i+1,j]-2.0*pcur[i,j]+pcur[i-1,j]            
            d2pdz2 = pcur[i,j+1]-2.0*pcur[i,j]+pcur[i,j-1]
            
            pnew[i,j] = 2.0*pcur[i,j] -pold[i,j] + fact[i,j]*(d2pdx2 + d2pdz2)
        end
    end

    
    ##====================================================    
    # inject source(s)
     for l=1:size(dt2srctf,2)
        isrc = ijsrcs[l,1]
        jsrc = ijsrcs[l,2]
        pnew[isrc,jsrc] = pnew[isrc,jsrc] + dt2srctf[t,l]
    end
    
    # assign the new pold and pcur
    ## https://discourse.julialang.org/t/swap-array-contents/7774/7
    ## kind of swapping array pointers, so NEED TO RETURN NEW BINDINGS,
    ##  otherwise the exchange is lost!
    pold,pcur,pnew = pcur,pnew,pold

    ### this is slower
    #  pold .= pcur 
    #  pcur .= pnew

    return pold,pcur,pnew
end


################################################################################################
################################################################################################
################################################################################################


function oneiter_CPML!slow(nx::Int64,nz::Int64,fact::Array{Float64,2},pnew::Array{Float64,2},
                           pold::Array{Float64,2},pcur::Array{Float64,2},dt2srctf::Array{Float64,2},
                           psi_x::Array{Float64,2},psi_z::Array{Float64,2},
                           xi_x::Array{Float64,2},xi_z::Array{Float64,2},
                           cpml::CoefPML,ijsrcs::Array{Int64,2},t::Int64)


    ###############################
    ### SLOW but simple VERSION
    ###############################

    ## compute current psi_x and psi_z first (need derivatives next)
    for j = 2:nz-1 # 3:nz-2
        for i = 2:nx-1 # 3:nx-2
            dpdx = pcur[i+1,j]-pcur[i,j] 
            dpdz = pcur[i,j+1]-pcur[i,j]
            psi_x[i,j] = cpml.b_x_half[i] / cpml.K_x_half[i] * psi_x[i,j] + cpml.a_x_half[i] * dpdx
            psi_z[i,j] = cpml.b_z_half[j] / cpml.K_z_half[j] * psi_z[i,j] + cpml.a_z_half[j] * dpdz
        end
    end

    for j = 2:nz-1 # 3:nz-2
        for i = 2:nx-1 # 3:nx-2

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

    ##====================================================    
    # inject source(s)
    for l=1:size(dt2srctf,2)
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


########################################################################

function smoothgradient1shot!(grad::AbstractArray,ijsrcs::Array{Int64,2} ;
                              radiuspx::Integer=5)

    isr = ijsrcs[1]
    jsr = ijsrcs[2]
    nx,ny = size(grad)

    rmax = radiuspx
    imin = isr-radiuspx
    imax = isr+radiuspx
    jmin = jsr-radiuspx
    jmax = jsr+radiuspx

    for j=jmin:jmax
        for i=imin:imax
            # deal with the borders
            if i<1 || i>nx || j<1 || j>ny
                continue
            else
                # inverse of geometrical spreading
                r = sqrt(float(i-isr)^2+float(j-jsr)^2)
                if r<=rmax
                    # normalized inverse of geometrical spreading
                    att = r/rmax
                    grad[i,j] *= att
                end
            end
        end
    end

    return 
end

####################################################################
