
#------------------------------------------------------------------------
#
#    Copyright (C) 2017  Andrea Zunino 
#
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#------------------------------------------------------------------------
 
#######################################################################
#######################################################################

# -*- coding: utf-8 -*-

##==========================================================
module elasticwaveprop


export InpParam,RockProperties,gaussource1D,rickersource1D
export solveelastic2D_reflbound,solveelastic2D_CPML,MomentTensor

using SpecialFunctions
##===============================================

# immutable, so no further changes after assignment
immutable RockProperties
    lambda::Array{Float64,2}
    mu::Array{Float64,2}
    rho::Array{Float64,2}
end

# immutable, so no further changes after assignment
immutable InpParam
    ntimesteps::Int64
    nx::Int64
    nz::Int64
    dt::Float64
    dh::Float64
    #sourcetype::String
    savesnapshot::Bool
    snapevery::Int64
    seismogrkind::String
    leftbound::String
    rightbound::String
    bottombound::String
    topbound::String
    #InpParam(ntimesteps=1,nx=1,ny=1,dx=1.0,dy=1.0) = new(ntimesteps,nx,ny,dx,dy)
end

## to pre-calculate coefficients for interpolation at receivers
type RecInterp
    x::Float64
    z::Float64
    imin::Int64
    imax::Int64
    jmin::Int64
    jmax::Int64
    wind::Array{Float64,2} ## window for sinc interpolation    
end

type SrcInterp
    x::Float64
    z::Float64
    imin::Int64
    imax::Int64
    jmin::Int64
    jmax::Int64
    wind::Array{Float64,2} ## window for sinc interpolation    
end

immutable MomentTensor
  Mxx::Float64
  Mzz::Float64
  Mxz::Float64
end

##=====================================================================
##=====================================================================

# kaiser window parameterized by alpha
function kaiser(n::Integer, alpha::Real)
    pf = 1.0/besseli(0,pi*alpha)
    kai = [pf*besseli(0, pi*alpha*(sqrt.(1 - (2*k/(n-1) - 1)^2))) for k=0:(n-1)]
    return kai
end

##=====================================================================

function kaiser(x::Vector, x0::Real, b::Real, r::Real)
    # Kaiser window function
    #  r is window half with
    # Rule of thumb for finite diff.:
    #   b=4.14  b=6.31
    #   r = 4.0*dx
    w=zeros(length(x))
    for i=1:length(x)
        if -r<=(x[i]-x0)<=r 
            den = 1.0/besseli(0,b)
            w[i] = den*besseli(0, b*(sqrt.(1 -((x[i]-x0)/r)^2)))
        else
            w[i] = 0.0
        end
    end
    return w
end

##=====================================================================

# function discrDeltaf(x::Vector,xsource::Real,dx::Real; derivorder::Integer=0)
#     # discrete delta function
#     d=zeros(length(x))
#     kmax = pi/dx # Fichtner 2010 book
#     if derivorder==0
#         for i=1:length(x)
#             ## Fichtner 2010 book eq 3.98:
#             ## d[i]=(kmax/pi)*sinc(kmax*(x[i]-xsource))
#             #
#             ## Hicks 2002 Geophysics   
#             d[i]=sinc(kmax*(x[i]-xsource))
#         end
#     elseif derivorder==1
#         for i=1:length(x)
#             ## Hicks 2002 Geophysics
#             tmp = kmax*(x[i]-xsource)
#             numer = tmp*cos(tmp)-sin(tmp)
#             denom = tmp^2
#             if denom==0.0
#                 d[i]=0.0
#             else
#                 d[i] = kmax * numer / denom
#             end
#         end
#     else
#         return nothing
#     end
#     return d
# end

##=====================================================================

# function interprecs(xstart::Real,ystart::Real,dx::Real,dy::Real,
#                     xrec::Real,zrec::Real,fieldval::Array{Float64,2} )
    
#     # compute coefficients
#     #  ix and iz contain the indices i_min,i_max and j_min,j_max of the 2D window
#     xzwind,ix,iz = coeffsinc2D(xstart,zstart,dx,dz,xrec,zrec)

#     # get the interpolated value at rec position
#     recval = sum(fieldval[ix[1]:ix[2],iz[1]:iz[2]].*xzwind)
#     return recval
# end

##=====================================================================

function setupreceivinterp(xstart::Real,zstart::Real,dx::Real,dz::Real,
                        nx::Int64,nz::Int64,recpos::Array{Float64,2})

    # compute coefficients
    #  ix and iz contain the indices i_min,i_max and j_min,j_max of the 2D window

    nrecs = size(recpos,1)
    recint = Array{Any}(nrecs)
    for r=1:nrecs
        ## compute window
        xzwind,ix,iz = coeffsinc2D(xstart,zstart,dx,dz,recpos[r,1],recpos[r,2])
        ## check valid position 
        if ix[1]<1
            println("setuprecinter(): Error receiver too close to boundaries! Quitting.")
            error("setuprecinter()")
        elseif ix[2]>nx
            println("setuprecinter(): Error receiver too close to boundaries! Quitting.")
             error("setuprecinter()")
        elseif iz[1]<1
            println("setuprecinter(): Receiver close to top boundary, using mirroring technique.")
            println("setuprecinter():    See Hicks 2002, Geophysics.")
            println("setuprecinter():  To be tested!!!")
            ##-------------------------
            nn = -iz[1]+2
            # using less points for sinc... to be tested!!
            tmp = xzwind[:,nn:end]
            tmp2 = tmp[:,1:nn-1] + xzwind[:,nn-1:-1:1]
            tmp[:,1:nn-1] = tmp2
            xzwind = tmp
            iz[1] = 1
        elseif iz[2]>nz
            println("setuprecinter(): Error receiver too close to boundaries! Quitting.")
             error("setuprecinter()")
        end
        #-------------
        recint[r] = RecInterp(recpos[r,1],recpos[r,2],ix[1],ix[2],iz[1],iz[2],xzwind)
    end
    return recint
end

##=====================================================================

function setupsourceinterp(xstart::Real,zstart::Real,dx::Real,dz::Real,
                           nx::Int64,nz::Int64,srcpos::Array{Float64,2},
                           deriv_x=false, deriv_z=false )

    # compute coefficients
    #  ix and iz contain the indices i_min,i_max and j_min,j_max of the 2D window

    nsrc = size(srcpos,1)
    if nsrc>1
        error("Only one source allowed for now...")
    end

 
    srcint = Array{Any}(nsrc)
    for r=1:nsrc
        ## compute sinc window for each moment tensor component
        ## xstart+dx/2 staggered grid
        xzwind,ix,iz = coeffsinc2D(xstart,zstart,dx,dz,srcpos[r,1],srcpos[r,2] ;
                                   deriv_x=deriv_x, deriv_y=deriv_z )
          
        ## check valid position 
        if ix[1]<1
            println("setupsrcinter(): Error source too close to boundaries! Quitting.")
            error("setupsrcinter()")
        elseif ix[2]>nx
            println("setupsrcinter(): Error source too close to boundaries! Quitting.")
            error("setupsrcinter()")
        elseif iz[1]<1
            println("setupsrcinter(): Source close to free surface, using mirroring technique.")
            println("setupsrcinter():    See Hicks 2002, Geophysics.")
            println("setupsrcinter():  To be tested!!!")
            ##-------------------------
            nn = -iz[1]+2
            # using less points for sinc... to be tested!!
            tmp = xzwind[:,nn:end]
            tmp2 = tmp[:,1:nn-1] + xzwind[:,nn-1:-1:1]
            tmp[:,1:nn-1] = tmp2
            xzwind = tmp
            iz[1] = 1
        elseif iz[2]>nz
            println("setupsrcinter(): Error source too close to boundaries! Quitting.")
            error("setupsrcinter()")
        end
        #-------------
        srcint[r] = SrcInterp(srcpos[r,1],srcpos[r,2],ix[1],ix[2],iz[1],iz[2],xzwind)
    end
    return srcint
end

##=====================================================================

function coeffsinc(xstart::Real,dx::Real,xcenter::Real; deriv=false,
                   npts::Int64=4, beta::Real=4.14)
    ## Coefficients for sinc interpolation
    ##  in 1D
    ##  xstart is the x coordinate of first node in the regular grid    
    ##  
    ## beta = 4.14 Hicks 2002, Geophysics
    ###
    ### Julia:  sinc(x) =
    ###    \sin(\pi x) / (\pi x) if x \neq 0, and 1 if x = 0
    ###
    rx = npts*dx
    ## Assuming x from grid starts xstart
    xh = (xcenter-xstart)/dx
    ix = floor(Int64,xh+1)
    if mod((xcenter-xstart),dx) == 0.0
        ixsta = ix-npts
        ixend = ix+npts
    else
        ixsta = ix-npts+1
        ixend = ix+npts
    end
    x = [xstart+dx*(i-1) for i=ixsta:ixend]
    
    if deriv==false
        # interpolating sinc(x) = sin(pi*x)/(pi*x) see Julia definition
        intrpsinc = sinc.((x-xcenter)/dx)
        
    elseif deriv==true
        # derivative 
        intrpsinc = zeros(length(x))
        for i=1:length(x)
            ## Hicks 2002 Geophysics
            tmp = kmax*(x[i]-xsource)
            numer = tmp*cos(tmp)-sin(tmp)
            denom = tmp^2
            if denom==0.0
                intrpsinc[i]=0.0
            else
                intrpsinc[i] = kmax * numer / denom
            end
        end
    end
    # apply Kaiser windowing
    kaix = kaiser(x,xcenter,beta,rx)
    windx = kaix.*intrpsinc
    # return also indices of window
    return windx,[ixsta,ixend]
end

##==================================================================

function coeffsinc2D(xstart::Real,ystart::Real,dx::Real,dy::Real,
                     xcenter::Real,ycenter::Real ;
                     deriv_x::Bool=false, deriv_y::Bool=false, 
                     npts::Int64=4, beta::Real=4.14)
    ## Calculate the 2D array of coefficients
    windx,ixstaend = coeffsinc(xstart,dx,xcenter; deriv=deriv_x, npts=4, beta=4.14)
    windy,iystaend = coeffsinc(ystart,dy,ycenter; deriv=deriv_y, npts=4, beta=4.14)
    # tensor product of x and y
    xywind = windx * windy' # transposed, outer prod
    # return also indices of window
    return xywind,ixstaend,iystaend
end

##=====================================================================

function cumsum(x, y)
    # Trapezoidal integration rule
    n = length(x)
    assert(n==length(y))
    if n == 1
        r = 0.0
        return r
    end
    r = zeros(Float64,n)
    for i=2:n-1
        r[i] = r[i-1] + (x[i] - x[i-1]) * (y[i] + y[i-1])
    end
    r = r/2.0
    return r
end

##======================================================

function trapz(x, y)
    # Trapezoidal integration rule
    n = length(x)
    assert(n==length(y))
    if n == 1
        r = 0.0
        return r
    end
    r = 0.0
    for i=1:n-1
        r += (x[i+1] - x[i]) * (y[i+1] + y[i])
    end
    r = r/2.0
    return r
end

##======================================================

# function bilinear_interp(f::Array{Float64,2},hgrid::Float64, pt::Array{Float64,1})
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

##======================================================

function gaussource1D( t, t0::Float64, f0::Float64 )
    a = (pi*f0)^2
    source = - 8.0*a*(t-t0).*exp.( -a.*(t-t0).^2 )    
    return source
end

##============================================================

function rickersource1D( t::Array{Float64},  t0::Float64, f0::Float64 )    
    b = (pi*f0*(t-t0)).^2
    w = (1.-2.*b).*exp.(-b)
    return w
end   

##==================================================================


function calc_dKa_CPML(nptspml::Integer,gridspacing,dt,Npower,d0,
                       alpha_max_pml,K_max_pml,onwhere )

    # L = thickness of adsorbing layer
    if onwhere=="grdpts"
        L = nptspml*gridspacing
        # distances 
        x = collect(range(0.0,gridspacing,nptspml))
    elseif onwhere=="halfgrdpts"
        L = nptspml*gridspacing
        # distances 
        x = collect(range(gridspacing/2.0,gridspacing,nptspml))
    end
    
    d = d0 * (x./L).^Npower
    K = 1.0 + (K_max_pml - 1.0) * (x./L).^Npower
    #from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-2
    alpha =  alpha_max_pml * (1.0 - (x./L)) + 0.1 * alpha_max_pml
    b = exp.( - (d ./ K + alpha) .* dt )
    a = d .* (b-1.0)./(K.*(d+K.*alpha))
    
    return d,K,alpha,b,a
end


##==========================================================================

function solveelastic2D_CPML(inpar, rockprops, srcpos, sourcetf, momtens, srcdomfreq::Real,
                              recpos::Array{Float64,2})
    #  
    # Wave elastic staggered grid 2D solver 
    #
    # Andrea Zunino, July 3, 2017
    #
    # Staggered grid with equal spacing in x and z.
    # Second order in time, fourth order in space.
    # Convolutionary Perfectly Matched Layer (C-PML) boundary conditions
    #
    #   References:
    #       Levander A. (1988), Fourth-order finite-difference P-SV seismograms, Geophysics.
    #       Komatitsch D. and Martin R. (2007), An unsplit convolutional
    #            perfectly matched layer improved at grazing incidence for the
    #            seismic wave equation, Geophysics.
    #       Robertsson, J.O. (1996) Numerical Free-Surface Condition for
    #            Elastic/Viscoelastic Finite-Difference Modeling in the Presence
    #            of Topography, Geophysics.
    #
    #
    #   STAGGERED GRID 
    #                      x
    #     +---------------------------------------------------->
    #     |
    #     |
    #     |                (i)    (i+1/2)   (i+1)  (i+3/2)
    #     |                 |        |        |       |
    #     |                 |        |        |       |
    #  z  |       (j) ---vx,rho-----Txx----vx,rho -----
    #     |              lam,mu     Tzz    lam,mu     |
    #     |                 |        |        |       |
    #     |                 |        |        |       |
    #     |  (j+1/2)  -----Txz------vz-------Txz-------
    #     |                 |        |        |       |
    #     |                 |        |        |       |
    #     |                 |        |        |       |
    #     |    (j+1) ----vx,rho-----Txx-----vx,rho-----
    #     |              lam,mu     Tzz     lam,mu
    #     v
    #      
    #   Where
    #
    #   Txx Stress_xx (Normal stress x)
    #   Tzz: Stress_zz (Normal stress z)
    #   Txz: Stress_xz (Shear stress)
    #   vx: Velocity_x (x component of velocity) 
    #   vz: Velocity_z (z component of velocity) 
    #
    #
    # Node indexing:
    # ------------------------------------------------
    # |      Code         |   Staggered grid         |
    # ------------------------------------------------
    # | rho(i,j)          | rho(i,j)                 |
    # | lam(i,j),mu(i,j)  | lam(i,j),mu(i,j)         |
    # | Txx(i,j),Tzz(i,j) | Txx(i+1/2,j),Tzz(i+1/2,j)|
    # | Txz(i,j)          | Txz(i,j+1/2)             |
    # | vx(i,j)           | vx(i,j)                  | 
    # | vz(i,j)           | vz(i+1/2,j+1/2)          |
    # ------------------------------------------------
        
    ##############################  
    # Lame' parameters
    ##############################
    lambda = rockprops.lambda
    mu = rockprops.mu
    ##############################
    # Density
    ##############################
    rho = rockprops.rho

    #############################
    dh = inpar.dh
    dx = dh
    dz = dh
    dt = inpar.dt
    nx = inpar.nx
    nz = inpar.nz
    
    xmax = (nx-1)*dh
    zmax = (nz-1)*dh
    println("\n Size of grid: $xmax in x and $zmax in z")
    println(" Receivers position: $recpos ")
    println(" Source position: $srcpos ")
    
    ##############################
    ## Check stability criterion
    ##############################
    maxvp = maximum( sqrt.( (lambda+2.0*mu)./rho ))
    Courant_number = maxvp * inpar.dt * sqrt.(1.0/dx^2 + 1.0/dz^2)
    if Courant_number > 1.0
        println(" The stability criterion is violated. Quitting.")
        error("The stability criterion is violated.")
    end

    ###############################
    ## Check dispersion criterium
    ###############################
    fmaxsource = srcdomfreq # very approx.... # Nyquist: 1/(2.0*dt)???
    arbfact = 1.0 # * max freq to be safe... boh?
    ngridptsperwavlen = 8 # standard value for this setup
    minvs = minimum( sqrt.(mu./rho) )
    maxalloweddh = minvs/(ngridptsperwavlen*arbfact*fmaxsource)
    if dh>maxalloweddh 
        println(" The dispersion criterion is violated. Quitting.")
        error("The dispersion criterion is violated.")
    end

    ##############################
    #   Parameters
    ##############################

    harmonicaver_mu = false
    
    saveevery = inpar.snapevery
    f0 = srcdomfreq # source

    ## pml = c-pml, else reflsurf
    ## freesur = free surf for top
    topbound    = inpar.topbound
    leftbound   = inpar.leftbound
    rightbound  = inpar.rightbound
    bottombound = inpar.bottombound
    @assert (topbound in ["freesurf","pml","reflsurf"])
    if topbound=="freesurf"
        freeboundtop=true
    else
        freeboundtop=false
    end
    @assert (leftbound in ["pml","reflsurf"])
    @assert (rightbound in ["pml","reflsurf"])
    @assert (bottombound in ["pml","reflsurf"])

    println("Boundary conditions: ")
    println("   |------- $topbound -------| ")
    println("   |                        |")
    println(" $leftbound               $rightbound")
    println("   |                        | ")
    println("   |------- $bottombound -------|")

    
    ##############################
    #   Parameters for PML
    ##############################
    nptspml_x = convert(Int64,ceil(1.2*(maxvp/f0)/dx))  
    nptspml_z = convert(Int64,ceil(1.2*(maxvp/f0)/dz))  

    println(" Size of PML layers in grid points: $nptspml_x in x and $nptspml_z in z")
    
    Npower::Float64 = 2.0
    @assert Npower >= 1
    
    K_max_pml::Float64 = 1.0
    alpha_max_pml::Float64 = 2.0*pi*(f0/2.0) 
    
    # reflection coefficient
    Rcoef::Float64 = 0.001
       
    # thickness of the PML layer in meters
    thickness_pml_x::Float64 = nptspml_x * dx
    thickness_pml_z::Float64 = nptspml_z * dz
    
    # compute d0 
    d0_x::Float64 = - (Npower + 1) * maxvp * log(Rcoef) / (2.0 * thickness_pml_x)
    d0_z::Float64 = - (Npower + 1) * maxvp * log(Rcoef) / (2.0 * thickness_pml_z)
    
    
    ##############################
    #   Damping parameters
    ##############################
    
    # --- damping in the x direction ---
    # assuming the number of grid points for PML is the same on 
    #    both sides    
    # damping profile at the grid points
    d_xpml,K_xpml,alpha_xpml,b_xpml,a_xpml = calc_dKa_CPML(nptspml_x,dh,dt,Npower,d0_x,alpha_max_pml,K_max_pml,"grdpts")
    # damping profile at half the grid points
    d_xpml_half,K_xpml_half,alpha_xpml_half,b_xpml_half,a_xpml_half = calc_dKa_CPML(nptspml_x,dh,dt,Npower,d0_x,alpha_max_pml,K_max_pml,"halfgrdpts")
    
    # --- damping in the z direction ---
    # assuming the number of grid points for PML is the same on
    # both sides    
    # damping profile at the grid points
    d_zpml,K_zpml,alpha_zpml,b_zpml,a_zpml = calc_dKa_CPML(nptspml_z,dh,dt,Npower,d0_z,alpha_max_pml,K_max_pml,"grdpts")
    # damping profile at half the grid points
    d_zpml_half,K_zpml_half,alpha_zpml_half,b_zpml_half,a_zpml_half = calc_dKa_CPML(nptspml_z,dh,dt,Npower,d0_z,alpha_max_pml,K_max_pml,"halfgrdpts")
    
    #######################
    #  x direction
    #######################
    d_x     = zeros(Float64,nx)
    K_x     = ones(Float64,nx)
    alpha_x = zeros(Float64,nx)
    b_x     = zeros(Float64,nx)
    a_x     = zeros(Float64,nx)
    d_x_half     = zeros(Float64,nx)
    K_x_half     = ones(Float64,nx)
    alpha_x_half = zeros(Float64,nx)
    b_x_half     = zeros(Float64,nx)
    a_x_half     = zeros(Float64,nx)

    if leftbound=="pml"
        # # reverse coefficients to get increasing damping away from inner model
        # left boundary
        d_x[1:nptspml_x]     = d_xpml[end:-1:1]
        K_x[1:nptspml_x]     = K_xpml[end:-1:1]
        alpha_x[1:nptspml_x] = alpha_xpml[end:-1:1]
        b_x[1:nptspml_x]     = b_xpml[end:-1:1]
        a_x[1:nptspml_x]     = a_xpml[end:-1:1]
        # half stuff...
        # reverse coefficients to get increasing damping away from inner model
        # left boundary
        #  One less element on left boundary (end-1...)
        #    because of the staggered grid 
        d_x_half[1:nptspml_x-1]     = d_xpml_half[end-1:-1:1]
        K_x_half[1:nptspml_x-1]     = K_xpml_half[end-1:-1:1]
        alpha_x_half[1:nptspml_x-1] = alpha_xpml_half[end-1:-1:1]
        b_x_half[1:nptspml_x-1]     = b_xpml_half[end-1:-1:1]
        a_x_half[1:nptspml_x-1]     = a_xpml_half[end-1:-1:1]

    end
    if rightbound=="pml"
        # right boundary
        rightpml = nx-nptspml_x+1 
        d_x[rightpml:end]     = d_xpml
        K_x[rightpml:end]     = K_xpml
        alpha_x[rightpml:end] = alpha_xpml
        b_x[rightpml:end]     = b_xpml
        a_x[rightpml:end]     = a_xpml 
        #half stuff 
        # right boundary
        d_x_half[rightpml:end]     = d_xpml_half
        K_x_half[rightpml:end]     = K_xpml_half
        alpha_x_half[rightpml:end] = alpha_xpml_half
        b_x_half[rightpml:end]     = b_xpml_half
        a_x_half[rightpml:end]     = a_xpml_half 

    end
    
    #######################
    #  z direction
    #######################
    d_z     = zeros(Float64,nz)
    K_z     = ones(Float64,nz)
    alpha_z = zeros(Float64,nz)
    b_z     = zeros(Float64,nz)
    a_z     = zeros(Float64,nz)
    d_z_half     = zeros(Float64,nz)
    K_z_half     = ones(Float64,nz)
    alpha_z_half = zeros(Float64,nz)
    b_z_half     = zeros(Float64,nz)
    a_z_half     = zeros(Float64,nz)

    if bottombound=="pml"
        # bottom 
        bottompml=nz-nptspml_z+1
        d_z[bottompml:end]     = d_zpml
        K_z[bottompml:end]     = K_zpml
        alpha_z[bottompml:end] = alpha_zpml
        b_z[bottompml:end]     = b_zpml
        a_z[bottompml:end]     = a_zpml
        # half stuff...
        # reverse coefficients to get increasing damping away from inner model  
        # bottom
        d_z_half[bottompml:end]     = d_zpml_half
        K_z_half[bottompml:end]     = K_zpml_half
        alpha_z_half[bottompml:end] = alpha_zpml_half
        b_z_half[bottompml:end]     = b_zpml_half
        a_z_half[bottompml:end]     = a_zpml_half
    end
    # if PML also on top of model...
    if topbound=="pml"
        # on grid
        d_z[1:nptspml_z]     = d_zpml[end:-1:1]
        K_z[1:nptspml_z]     = K_zpml[end:-1:1]
        alpha_z[1:nptspml_z] = alpha_zpml[end:-1:1]
        b_z[1:nptspml_z]     = b_zpml[end:-1:1]
        a_z[1:nptspml_z]     = a_zpml[end:-1:1]
        # half
        #  One less element on top boundary (end-1...)
        #    because of the staggered grid
        d_z_half[1:nptspml_z-1]     = d_zpml_half[end-1:-1:1]
        K_z_half[1:nptspml_z-1]     = K_zpml_half[end-1:-1:1]
        alpha_z_half[1:nptspml_z-1] = alpha_zpml_half[end-1:-1:1]
        b_z_half[1:nptspml_z-1]     = b_zpml_half[end-1:-1:1]
        a_z_half[1:nptspml_z-1]     = a_zpml_half[end-1:-1:1]
    end

    #####################################################

    ## Arrays to export snapshots
    if inpar.savesnapshot==true
        ntsave = convert(Int64,div(inpar.ntimesteps,saveevery))
        vxsave = zeros(Float64,inpar.nx,inpar.nz,ntsave)
        vzsave = zeros(Float64,inpar.nx,inpar.nz,ntsave)
        tsave=1
    end

    ## Arrays to return seismograms
    nrecs = size(recpos,1)
    receiv = zeros(Float64,inpar.ntimesteps,nrecs,2)
    
    ########################################
    ##  Setup interpolation for receivers  #
    ########################################
    ## separate for vx and vz because of the staggered grid,
    ##   so  xstart,ystart is different,
    ## vx is on integer grid and vz on half,half grid
    reit_vx = setupreceivinterp(0.0,   0.0,   dx,dz,nx,nz,recpos)
    reit_vz = setupreceivinterp(dx/2.0,dz/2.0,dx,dz,nx,nz,recpos)


    ########################################
    ##  Setup interpolation for sources    #
    ########################################
    ## separate for vx and vz because of the staggered grid,
    ##   so  xstart,ystart is different,
    ## vx is on integer grid and vz on half,half grid
    srcit_Mxx = setupsourceinterp(dx/2, 0.0,  dx,dz,nx,nz,srcpos,false, false)
    srcit_Mzz = setupsourceinterp(dx/2, 0.0,  dx,dz,nx,nz,srcpos,false, false)
    srcit_Mxz = setupsourceinterp(0.0,  dz/2, dx,dz,nx,nz,srcpos,false, false)
    nsources = size(srcpos,1)
        
    #########################
    # Source time function
    lensrctf = length(sourcetf)
    #isrc = ijsrc[1]
    #jsrc = ijsrc[2]
    # isrc = ijsrc[1]+nptspml_x
    # if freeboundtop==true
    #     jsrc = ijsrc[2]
    # else
    #     jsrc = ijsrc[2]+nptspml_y
    # end
    
    ## Initialize arrays
    vx = zeros(Float64,nx,nz)
    vz = zeros(Float64,nx,nz)
    Txx = zeros(Float64,nx,nz)
    Tzz = zeros(Float64,nx,nz)
    Txz = zeros(Float64,nx,nz)

    # PML arrays
    # Arrays with size of PML areas would be sufficient and save memory,
    #   however allocating arrays with same size than model simplifies
    #   the code in the loops
    psi_DxTxx = zeros(Float64,nx,nz)
    psi_DzTxz = zeros(Float64,nx,nz)
    
    psi_DxTxz = zeros(Float64,nx,nz)
    psi_DzTzz = zeros(Float64,nx,nz)
    
    psi_DxVx = zeros(Float64,nx,nz)
    psi_DzVz = zeros(Float64,nx,nz)

    psi_DzVx = zeros(Float64,nx,nz)
    psi_DxVz = zeros(Float64,nx,nz)

    
    ##############################
    #   Derivative operators
    ##############################
    #
    #  o => point where to take the derivative
    #
    # forward operator: 1/24 * (f[i-1]-27*f[i]+27*f[i+1]-f[i+2])
    #
    #  i-1   i    i+1  i+2
    #   |    |  o  |    |
    #
    # backward operator: 1/24 * (f[i-2]-27*f[i-1]+27*f[i]-f[i+1])
    #
    #  i-2  i-1    i   i+1
    #   |    |  o  |    |
    #    
    # Weigths for taking derivarives for incresing indices
    #Dweights = 1.0/inpar.dh * [1/24.0, -27.0/24.0, 27.0/24.0, -1/24.0]

    fact::Float64 = 1.0/(24.0*dh)
    
    ###############################################################
    # pre-interpolate properties at half distances between nodes
    ###############################################################
    # rho_ihalf_jhalf (nx-1,ny-1) ??
    rho_ihalf_jhalf = (rho[2:end,2:end]+rho[2:end,1:end-1]+rho[1:end-1,2:end]+rho[1:end-1,1:end-1])/4.0
    # mu_ihalf (nx-1,ny) ??
    # mu_ihalf (nx,ny-1) ??
    if harmonicaver_mu==true 
        # harmonic mean
        mu_ihalf = 1.0./( 1.0./mu[2:end,:] + 1.0./mu[1:end-1,:] )
        mu_jhalf = 1.0./( 1.0./mu[:,2:end] + 1.0./mu[:,1:end-1] )
    else
        mu_ihalf = (mu[2:end,:]+mu[1:end-1,:])/2.0 ###?????
        mu_jhalf = (mu[:,2:end]+mu[:,1:end-1])/2.0 ###?????
    end
    # lambda_ihalf (nx-1,ny) ??
    lambda_ihalf = (lambda[2:end,:]+lambda[1:end-1,:])/2.0 ###?????

    ##======================================##
    Dxb_Txx::Float64 = 0.0
    Dzb_Txz::Float64 = 0.0
    Dxf_Txz::Float64 = 0.0
    Dzf_Tzz::Float64 = 0.0
    Dxf_vx::Float64 = 0.0
    Dzb_vz::Float64 = 0.0
    Dzf_vx::Float64 = 0.0
    Dxb_vz::Float64 = 0.0
    
    ##======================================##
    ## time loop
    dt = inpar.dt
    println(" Time step: $dt ")
    starttime=time()

    for t=1:inpar.ntimesteps        

        if mod(t,50)==0
            curtime=(time()-starttime)
            print("\r Time step $t of $(inpar.ntimesteps)  $curtime s")
        end
        
        ## Inject the source as stress from moment tensor
        ##  See Igel 2017 Computational Seismology (book) page 31, 2.6.1
        if t<=lensrctf

            for s=1:nsources
                ## interpolate source at nearby nodes using sinc method
                Txx[srcit_Mxx[s].imin:srcit_Mxx[s].imax,srcit_Mxx[s].jmin:srcit_Mxx[s].jmax] =
                    Txx[srcit_Mxx[s].imin:srcit_Mxx[s].imax,srcit_Mxx[s].jmin:srcit_Mxx[s].jmax] +
                    (srcit_Mxx[s].wind .* momtens.Mxx) .* sourcetf[t] * dt 
                Tzz[srcit_Mzz[s].imin:srcit_Mzz[s].imax,srcit_Mzz[s].jmin:srcit_Mzz[s].jmax] =
                    Tzz[srcit_Mzz[s].imin:srcit_Mzz[s].imax,srcit_Mzz[s].jmin:srcit_Mzz[s].jmax] +
                    (srcit_Mzz[s].wind .* momtens.Mzz) .* sourcetf[t] * dt 
                Txz[srcit_Mxz[s].imin:srcit_Mxz[s].imax,srcit_Mxz[s].jmin:srcit_Mxz[s].jmax] =
                    Txz[srcit_Mxz[s].imin:srcit_Mxz[s].imax,srcit_Mxz[s].jmin:srcit_Mxz[s].jmax] +
                    (srcit_Mxz[s].wind .* momtens.Mxz) .* sourcetf[t] * dt 
            end

            
            ## only for testing !!!
            # xh=srcit_Mxx[1].x/dh
            # yh=srcit_Mxx[1].z/dh
            # isrc=floor(Int64,xh+1) # indices starts from 1
            # jsrc=floor(Int64,yh+1) # indices starts from 1
            # t==1 && println("position of source in i,j: $isrc $jsrc, x,y: $(srcit_Mxx[1].x) $(srcit_Mxx[1].z)")
            # Txx[isrc,jsrc] = Txx[isrc,jsrc] + momtens.Mxx * sourcetf[t]* dt 
            # Tzz[isrc,jsrc] = Tzz[isrc,jsrc] + momtens.Mzz * sourcetf[t]* dt 
            # Txz[isrc,jsrc] = Txz[isrc,jsrc] + momtens.Mxz * sourcetf[t]* dt

            
        end  
        
        ## space loops excluding boundaries

        #########################################
        # update velocities from stresses
        #########################################

        ###############################################
        #   Free surface boundary condition
        ## from Robertsson 1996
        if freeboundtop==true
            for j = 1:2
                for i = 3:nx-1
                    
                    # Vx
                    # Txx derivative only in x so no problem
                    Dxb_Txx = fact * ( Txx[i-2,j] -27.0*Txx[i-1,j] +27.0*Txx[i,j] -Txx[i+1,j] )
                    # image, mirroring Txz[i,j-2] = -Txz[i,j+1], etc.
                    Dzb_Txz = fact * ( -Txz[i,j+1] +27.0*Txz[i,j] +27.0*Txz[i,j] -Txz[i,j+1] )
                    # update velocity
                    vx[i,j] = vx[i,j] + (dt/rho[i,j]) * (Dxb_Txx + Dzb_Txz)

                end
            end
            
            for j = 1:2         
                for i = 2:nx-2     

                    # Vz
                    # Txz derivative only in x so no problem
                    Dxf_Txz = fact * ( Txz[i-1,j] -27.0*Txz[i,j] +27.0*Txz[i+1,j] -Txz[i+2,j] )
                    # image, mirroring Tzz[i,j-1] = -Txz[i,j+2], etc.
                    Dzf_Tzz = fact * ( -Tzz[i,j+2] +27.0*Tzz[i,j+1] +27.0*Tzz[i,j+1] -Tzz[i,j+2] )
                    # update velocity (rho has been interpolated in advance)
                    vz[i,j] = vz[i,j] + (dt/rho_ihalf_jhalf[i,j]) * (Dxf_Txz + Dzf_Tzz)
                    
                end
            end
        end # end free surf

        #########################################
        #  vx
        for j = 3:nz-1
            for i = 3:nx-1        
                
                # Vx
                Dxb_Txx = fact * ( Txx[i-2,j] -27.0*Txx[i-1,j] +27.0*Txx[i,j] -Txx[i+1,j] )
                Dzb_Txz = fact * ( Txz[i,j-2] -27.0*Txz[i,j-1] +27.0*Txz[i,j] -Txz[i,j+1] )

                # C-PML stuff 
                psi_DxTxx[i,j] = b_x[i] * psi_DxTxx[i,j] + a_x[i]*Dxb_Txx
                psi_DzTxz[i,j] = b_z[j] * psi_DzTxz[i,j] + a_z[j]*Dzb_Txz

                Dxb_Txx = Dxb_Txx / K_x[i] + psi_DxTxx[i,j]
                Dzb_Txz = Dzb_Txz / K_z[j] + psi_DzTxz[i,j]
                
                # update velocity
                vx[i,j] = vx[i,j] + (dt/rho[i,j]) * (Dxb_Txx + Dzb_Txz)
                
            end
        end
        
        #########################################
        #  vz       
        for j = 2:nz-2
            for i = 2:nx-2
                
                # Vz
                Dxf_Txz = fact * ( Txz[i-1,j] -27.0*Txz[i,j] +27.0*Txz[i+1,j] -Txz[i+2,j] )
                Dzf_Tzz = fact * ( Tzz[i,j-1] -27.0*Tzz[i,j] +27.0*Tzz[i,j+1] -Tzz[i,j+2] )
                
                # C-PML stuff 
                psi_DxTxz[i,j] = b_x_half[i] * psi_DxTxz[i,j] + a_x_half[i]*Dxf_Txz
                psi_DzTzz[i,j] = b_z_half[j] * psi_DzTzz[i,j] + a_z_half[j]*Dzf_Tzz
                
                Dxf_Txz = Dxf_Txz / K_x_half[i] + psi_DxTxz[i,j]
                Dzf_Tzz = Dzf_Tzz / K_z_half[j] + psi_DzTzz[i,j]

                # update velocity (rho has been interpolated in advance)
                vz[i,j] = vz[i,j] + (dt/rho_ihalf_jhalf[i,j]) * (Dxf_Txz + Dzf_Tzz)
                
            end
        end
        

        #########################################
        # update stresses from velocities 
        #########################################

        ###############################################
        #   Free surface boundary condition
        ## from Robertsson 1996
        if freeboundtop==true
            ################################
            # Txx, Tzz
            # j=1: we are on the free surface!
            j = 1  
            for i = 2:nx-2                
                # Txx
                # vx derivative only in x so no problem
                Dxf_vx = fact * ( vx[i-1,j] -27.0*vx[i,j] +27.0*vx[i+1,j] -vx[i+2,j] )
                # using boundary condition to calculate Dzb_vz from Dxf_vx
                Dzb_vz = -(1.0-2.0*mu_ihalf[i,j]/lambda_ihalf[i,j])*Dxf_vx
                # Txx
                Txx[i,j] = Txx[i,j] + (lambda_ihalf[i,j]+2.0*mu_ihalf[i,j]) * dt * Dxf_vx +
                    lambda_ihalf[i,j] * dt * Dzb_vz
                
                # Tzz
                Tzz[i,j] = 0.0 # we are on the free surface!
            end
            
            # j=2: we are just below the surface (1/2)
            j = 2
            for i = 2:nx-2  
                # Txx
                # vx derivative only in x so no problem
                Dxf_vx = fact * ( vx[i-1,j] -27.0*vx[i,j] +27.0*vx[i+1,j] -vx[i+2,j] )
                # zero velocity above the free surface
                Dzb_vz = fact * ( 0.0 -27.0*vz[i,j-1] +27.0*vz[i,j] -vz[i,j+1] )
                # Txx
                Txx[i,j] = Txx[i,j] + (lambda_ihalf[i,j]+2.0*mu_ihalf[i,j]) * dt * Dxf_vx +
                    lambda_ihalf[i,j] * dt * Dzb_vz
                
                # Tzz
                Tzz[i,j] = Tzz[i,j] + (lambda_ihalf[i,j]+2.0*mu_ihalf[i,j]) * dt* Dzb_vz +
                    lambda_ihalf[i,j] * dt * Dxf_vx

            end
            
            ################################
            # Txz
            j = 1
            for i=3:nx-1    
                # zero velocity above the free surface
                Dzf_vx = fact * ( 0.0 -27.0*vx[i,j] +27.0*vx[i,j+1] -vx[i,j+2] )
                # vz derivative only in x so no problem
                Dxb_vz = fact * ( vz[i-2,j] -27.0*vz[i-1,j] +27.0*vz[i,j] -vz[i+1,j] )
                
                # Txz
                Txz[i,j] = Txz[i,j] + mu_jhalf[i,j] * dt * (Dzf_vx + Dxb_vz)

            end

        end # end free surf

        #########################################
        #  Txx, Tzz 
        for j = 3:nz-1
            for i = 2:nx-2                
                
                # Txx,Tzz
                Dxf_vx = fact * ( vx[i-1,j] -27.0*vx[i,j] +27.0*vx[i+1,j] -vx[i+2,j] )
                Dzb_vz = fact * ( vz[i,j-2] -27.0*vz[i,j-1] +27.0*vz[i,j] -vz[i,j+1] )
                
                # C-PML stuff 
                psi_DxVx[i,j] = b_x_half[i] * psi_DxVx[i,j] + a_x_half[i]*Dxf_vx
                psi_DzVz[i,j] = b_z[j] * psi_DzVz[i,j] + a_z[j]*Dzb_vz

                Dxf_vx = Dxf_vx / K_x_half[i] + psi_DxVx[i,j]
                Dzb_vz = Dzb_vz / K_z[j] + psi_DzVz[i,j]
                
                # Txx
                Txx[i,j] = Txx[i,j] + (lambda_ihalf[i,j]+2.0*mu_ihalf[i,j]) * dt * Dxf_vx +
                    lambda_ihalf[i,j] * dt * Dzb_vz

                ## derivatives are the same than for Txx
                # Tzz
                Tzz[i,j] = Tzz[i,j] + (lambda_ihalf[i,j]+2.0*mu_ihalf[i,j]) * dt* Dzb_vz +
                    lambda_ihalf[i,j] * dt * Dxf_vx
                
            end
        end
        
        #########################################
        #  Txz
        for j = 2:nz-2
            for i = 3:nx-1  

                # Txz
                Dzf_vx = fact * ( vx[i,j-1] -27.0*vx[i,j] +27.0*vx[i,j+1] -vx[i,j+2] )
                Dxb_vz = fact * ( vz[i-2,j] -27.0*vz[i-1,j] +27.0*vz[i,j] -vz[i+1,j] )

                # C-PML stuff 
                psi_DzVx[i,j] = b_z_half[j] * psi_DzVx[i,j] + a_z_half[j]*Dzf_vx
                psi_DxVz[i,j] = b_x[i] * psi_DxVz[i,j] + a_x[i]*Dxb_vz

                Dzf_vx = Dzf_vx / K_z[j] + psi_DzVx[i,j]
                Dxb_vz = Dxb_vz / K_x[i] + psi_DxVz[i,j]
                
                # Txz
                Txz[i,j] = Txz[i,j] + mu_jhalf[i,j] * dt * (Dzf_vx + Dxb_vz)

            end
        end

        # ########################################
        # ## from seismic_cpml (Komatitsch)
        # ## Dirichlet conditions (rigid boundaries) on the edges or at the bottom of the PML layers
        # vx[2,:] = 0.0
        # vx[nx-1,:] = 0.0        
        # vz[2,:] = 0.0
        # vx[:,nz-1] = 0.0        
        # vz[nx-1,:] = 0.0
        # vz[:,nz-1] = 0.0
        # if freeboundtop==false
        #     vx[:,2] = 0.0
        #     vz[:,2] = 0.0
        # end

        
        ########################################
        ##### receivers
          for r=1:nrecs
            #rec_vx = bilinear_interp(vx,dh,recpos[r,:])
            # ## vz at half grid points, so recpos - dh/2.0 in both x an z
            #rec_vz = bilinear_interp(vz,dh,recpos[r,:]- dh/2.0 ) # - dh/2.0)
            ##--------------------------------------------------
            ## interpolate seismograms using sinc method
            rec_vx=sum( reit_vx[r].wind .* vx[reit_vx[r].imin:reit_vx[r].imax,
                                              reit_vx[r].jmin:reit_vx[r].jmax] )
            rec_vz=sum( reit_vz[r].wind .* vz[reit_vz[r].imin:reit_vz[r].imax,
                                              reit_vz[r].jmin:reit_vz[r].jmax] )
            receiv[t,r,:] = [rec_vx rec_vz]
        end
        
        #### save snapshots
        if (inpar.savesnapshot==true) &  (mod(t,saveevery)==0)
            vxsave[:,:,tsave] = vx
            vzsave[:,:,tsave] = vz
            tsave=tsave+1
        end
        
    end
    ##### End time loop

    ##---------------------------------------
    ## Output seismograms
    println("\n Output seismogram kind: $(inpar.seismogrkind)")
    if inpar.seismogrkind=="displacement"
        tarr=collect(Float64,range(0.0,dt,inpar.ntimesteps))
        for r=1:nrecs
            receiv[:,r,1]=cumsum(tarr,receiv[:,r,1])  # x
            receiv[:,r,2]=cumsum(tarr,receiv[:,r,2])  # z
        end
    elseif inpar.seismogrkind=="velocity"
        nothing
    end
    

    println(" ")
    if (inpar.savesnapshot==true) 
        return receiv,vxsave,vzsave
    else
        return receiv
    end
end

##===================================================================
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
##===================================================================

# function solveelastic2D_reflbound(inpar, rockprops, ijsrc, sourcetf,recpos)
    
#     # Wave elastic staggered grid 2D solver 
#     #
#     # Andrea Zunino, 4/5-2017
#     #
#     # Staggered grid with equal spacing, A. Levander (1988)
#     # Second order time, fourth order space
#     #
#     #
#     #   STAGGERED GRID 
#     #
#     #                      x
#     #     +---------------------------------------------------->
#     #     |
#     #     |
#     #     |                (i)    (i+1/2)   (i+1)  (i+3/2)
#     #     |                 |        |        |       |
#     #     |                 |        |        |       |
#     #  z  |       (j) ---vx,rho-----Txx----vx,rho -----
#     #     |              lam,mu     Tzz    lam,mu     |
#     #     |                 |        |        |       |
#     #     |                 |        |        |       |
#     #     |  (j+1/2)  -----Txz------vz-------Txz-------
#     #     |                 |        |        |       |
#     #     |                 |        |        |       |
#     #     |                 |        |        |       |
#     #     |    (j+1) ----vx,rho-----Txx-----vx,rho-----
#     #     |              lam,mu     Tzz     lam,mu
#     #     v
#     #      
#     #   Where
#     #
#     #   Txx Stress_xx (Normal stress x)
#     #   Tzz: Stress_zz (Normal stress z)
#     #   Txz: Stress_xz (Shear stress)
#     #   vx: Velocity_x (x component of velocity) 
#     #   vz: Velocity_z (z component of velocity) 
#     #
#     #
#     # Node indexing:
#     # ------------------------------------------------
#     # |      Code         |   Staggered grid         |
#     # ------------------------------------------------
#     # | rho(i,j)          | rho(i,j)                 |
#     # | lam(i,j),mu(i,j)  | lam(i,j),mu(i,j)         |
#     # | Txx(i,j),Tzz(i,j) | Txx(i+1/2,j),Tzz(i+1/2,j)|
#     # | Txz(i,j)          | Txz(i,j+1/2)             |
#     # | vx(i,j)           | vx(i,j)                  | 
#     # | vz(i,j)           | vz(i+1/2,j+1/2)          |
#     # ------------------------------------------------


#     #####################################################
    
#     saveevery = inpar.snapevery
#     dh = inpar.dh

#     ## Arrays to export snapshots
#     if inpar.savesnapshot==true
#         ntsave = div(inpar.ntimesteps,saveevery)
#         vxsave = zeros(Float64,inpar.nx,inpar.nz,ntsave)
#         vzsave = zeros(Float64,inpar.nx,inpar.nz,ntsave)
#         tsave=1
#     end

#     ## Arrays to return seismograms
#     nrecs = size(recpos,1)
#     receiv = zeros(Float64,inpar.ntimesteps,nrecs,2)
    
#     #######################################################
        
#     # Lame' parameters
#     lambda = rockprops.lambda
#     mu = rockprops.mu
#     # Density
#     rho = rockprops.rho

#     ## Check stability criterion
#     maxvp = maximum( sqrt( (lambda+2.0*mu)./rho ))
#     if inpar.dt>= 0.606*(inpar.dh/maxvp)
#         println(" The stability criterion is violated. Quitting.")
#         return
#     end

#     # Source time function
#     lensrctf = length(sourcetf)
#     isrc = ijsrc[1]
#     jsrc = ijsrc[2]
    
#     ## Init
#     vx = zeros(inpar.nx,inpar.nz)
#     vz = zeros(inpar.nx,inpar.nz)
#     Txx = zeros(inpar.nx,inpar.nz)
#     Tzz = zeros(inpar.nx,inpar.nz)
#     Txz = zeros(inpar.nx,inpar.nz)
        
    
#     ##=====================================
#     # Derivative operators
#     #
#     #  o => point where to take the derivative
#     #
#     # forward operator: 1/24 * (f[i-1]-27*f[i]+27*f[i+1]-f[i+2])
#     #
#     #  i-1   i    i+1  i+2
#     #   |    |  o  |    |
#     #
#     # backward operator: 1/24 * (f[i-2]-27*f[i-1]+27*f[i]-f[i+1])
#     #
#     #  i-2  i-1    i   i+1
#     #   |    |  o  |    |
#     #    
#     # Weigths for taking derivarives for incresing indices
#     Dweights = 1.0/inpar.dh * [1/24.0, -27.0/24.0, 27.0/24.0, -1/24.0]

#     # pre-interpolate properties at half distances between nodes
#     # rho_ihalf_jhalf (nx-1,ny-1) ??
#     rho_ihalf_jhalf = (rho[2:end,2:end]+rho[2:end,1:end-1]+rho[1:end-1,2:end]+rho[1:end-1,1:end-1])/4.0
#     # mu_ihalf (nx-1,ny) ??
#     # harmonic mean
#     mu_ihalf = 1.0./( 1.0./mu[2:end,:] + 1.0./mu[1:end-1,:] )
#     # mu_ihalf = (mu[2:end,:]+mu[1:end-1,:])/2.0 ###?????
#     # mu_ihalf (nx,ny-1) ??
#     # harmonic mean
#     mu_jhalf = 1.0./( 1.0./mu[:,2:end] + 1.0./mu[:,1:end-1] )
#     # mu_jhalf = (mu[:,2:end]+mu[:,1:end-1])/2.0 ###?????
#     # lambda_ihalf (nx-1,ny) ??
#     lambda_ihalf = (lambda[2:end,:]+lambda[1:end-1,:])/2.0 ###?????
    
#     ##======================================##
#     ## time loop
#     dt = inpar.dt
#     for t=1:inpar.ntimesteps        

#         if mod(t,50)==0
#             print("\r time step $t of $(inpar.ntimesteps)")
#         end
        
#         ## Inject the source 
#         if t<=lensrctf
#             if inpar.sourcetype=="P"
#                 Txx[isrc,jsrc] = Txx[isrc,jsrc] + sourcetf[t]* dt 
#                 Tzz[isrc,jsrc] = Tzz[isrc,jsrc] + sourcetf[t]* dt 
#             elseif inpar.sourcetype=="S"
#                 Txz[isrc,jsrc] = Txz[isrc,jsrc] + sourcetf[t]* dt 
#             else
#                 println("Error source type badly defined.")
#                 return
#             end 
#         end  
                
#         ## space loops excluding boundaries

#         #########################################
#         # update velocities from stresses
#         #########################################
   
#         for j = 3:inpar.nz-1
#             for i = 3:inpar.nx-1        
#                 # update velocities from stresses
#                 # Vx
#                 Dxb_Txx = dot(Dweights,Txx[i-2:i+1,j])
#                 Dzb_Txz = dot(Dweights,Txz[i,      j-2:j+1])
#                 vx[i,j] = vx[i,j] + (dt/rho[i,j]) * (Dxb_Txx + Dzb_Txz)
#             end
#         end
        
#         for j = 2:inpar.nz-2
#             for i = 2:inpar.nx-2
#                 # Vz
#                 Dxf_Txz = dot(Dweights,Txz[i-1:i+2,j])
#                 Dzf_Tzz = dot(Dweights,Tzz[i,      j-1:j+2])
#                 vz[i,j] = vz[i,j] + (dt/rho_ihalf_jhalf[i,j]) * (Dxf_Txz + Dzf_Tzz)
#             end
#         end

#         #########################################
#         # update stresses from velocities
#         #########################################
        
#         for j = 3:inpar.nz-1
#             for i = 2:inpar.nx-2                
#                 # update stresses from velocities
#                 # Txx
#                 Dxf_vx = dot(Dweights,vx[i-1:i+2,j])
#                 Dzb_vz = dot(Dweights,vz[i,      j-2:j+1])   
#                 Txx[i,j] = Txx[i,j] + (lambda_ihalf[i,j]+2.0*mu_ihalf[i,j]) * dt * Dxf_vx +
#                     lambda_ihalf[i,j] * dt * Dzb_vz

#                 ## derivatives are the same than for Txx
#                 # Tzz
#                 Tzz[i,j] = Tzz[i,j] + (lambda_ihalf[i,j]+2.0*mu_ihalf[i,j]) * dt* Dzb_vz +
#                     lambda_ihalf[i,j] * dt * Dxf_vx 
#             end
#         end
        
#         for j = 2:inpar.nz-2
#             for i = 3:inpar.nx-1  
#                 # Txz
#                 Dzf_vx = dot(Dweights,vx[i,      j-1:j+2])
#                 Dxb_vz = dot(Dweights,vz[i-2:i+1,j])
#                 Txz[i,j] = Txz[i,j] + mu_jhalf[i,j] * dt * (Dzf_vx + Dxb_vz)
#             end
#         end

#         ##---------------------------------------
#         ## save seismograms
#         for r=1:nrecs
#             rec_vx = bilinear_interp(vx,dh,recpos[r,:])
#             ## vz at half grid points, so recpos - dh/2.0 in both x an z
#             rec_vz = bilinear_interp(vz,dh,recpos[r,:]- dh/2.0 ) # - dh/2.0)
#             receiv[t,r,:] = [rec_vx rec_vz]
#         end
        
#         # #### save snapshots
#         if  (inpar.savesnapshot==true) & (mod(t,saveevery)==0)
#             println("$(inpar.savesnapshot)")
#             vxsave[:,:,tsave] = vx
#             vzsave[:,:,tsave] = vz
#             tsave=tsave+1
#         end
        
#     end
#     ##### End time loop

#     println(" ")
#     if (inpar.savesnapshot==true) 
#         return receiv,vxsave,vzsave
#     else
#         return receiv
#     end
# end

##========================================


end ## module
##========================================================


