
module Elastic2D_Iso_CPML_Serial

# Dummy data module
module Data
Array = Base.Array
end

# the backend needs these
ones = Base.ones
zeros = Base.zeros



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
# Dweights = 1.0/inpar.dh * [1/24.0, -27.0/24.0, 27.0/24.0, -1/24.0]



function precomp_prop(ρ,μ,λ,dh)

   fact::Float64 = 1.0/(24.0*dh)
    
    #-------------------------------------------------------------
    # pre-interpolate properties at half distances between nodes
    #-------------------------------------------------------------
    # ρ_ihalf_jhalf (nx-1,ny-1) ??
    ρ_ihalf_jhalf = (ρ[2:end,2:end]+ρ[2:end,1:end-1]+ρ[1:end-1,2:end]+ρ[1:end-1,1:end-1])/4.0
    # μ_ihalf (nx-1,ny) ??
    # μ_ihalf (nx,ny-1) ??
    if harmonicaver_μ==true 
        # harmonic mean
        μ_ihalf = 1.0./( 1.0./μ[2:end,:] + 1.0./μ[1:end-1,:] )
        μ_jhalf = 1.0./( 1.0./μ[:,2:end] + 1.0./μ[:,1:end-1] )
    else
        μ_ihalf = (μ[2:end,:]+μ[1:end-1,:])/2.0 ###?????
        μ_jhalf = (μ[:,2:end]+μ[:,1:end-1])/2.0 ###?????
    end
    # λ_ihalf (nx-1,ny) ??
    λ_ihalf = (λ[2:end,:]+λ[1:end-1,:])/2.0 ###?????

    return μ_ihalf,μ_jhalf,λ_ihalf,fact
end


function update_vx!(vx,fact,σxx,σxz,dt,ρ,ψ_∂σxx∂x,ψ_∂σxz∂z,b_x,b_z,a_x,a_z,
                    freetop)

    if freeop
        for j = 1:2
            for i = 3:nx-1
                
                # Vx
                # σxx derivative only in x so no problem
                ∂σxx∂x_bkw = fact * ( σxx[i-2,j] -27.0*σxx[i-1,j] +27.0*σxx[i,j] -σxx[i+1,j] )
                # image, mirroring σxz[i,j-2] = -σxz[i,j+1], etc.
                ∂σxz∂z_bkw = fact * ( -σxz[i,j+1] +27.0*σxz[i,j] +27.0*σxz[i,j] -σxz[i,j+1] )
                # update velocity
                vx[i,j] = vx[i,j] + (dt/ρ[i,j]) * (∂σxx∂x_bkw + ∂σxz∂z_bkw)

            end
        end
    end
    
    #  vx
    for j = 3:nz-1
        for i = 3:nx-1        
            
            # Vx
            ∂σxx∂x_bkw = fact * ( σxx[i-2,j] -27.0*σxx[i-1,j] +27.0*σxx[i,j] -σxx[i+1,j] )
            ∂σxz∂z_bkw = fact * ( σxz[i,j-2] -27.0*σxz[i,j-1] +27.0*σxz[i,j] -σxz[i,j+1] )

            # C-PML stuff 
            ψ_∂σxx∂x[i,j] = b_x[i] * ψ_∂σxx∂x[i,j] + a_x[i] * ∂σxx∂x_bkw
            ψ_∂σxz∂z[i,j] = b_z[j] * ψ_∂σxz∂z[i,j] + a_z[j] * ∂σxz∂z_bkw

            ∂σxx∂x_bkw = ∂σxx∂x_bkw + ψ_∂σxx∂x[i,j]
            ∂σxz∂z_bkw = ∂σxz∂z_bkw + ψ_∂σxz∂z[i,j]

            # update velocity
            vx[i,j] = vx[i,j] + (dt/ρ[i,j]) * (∂σxx∂x_bkw + ∂σxz∂z_bkw)
            
        end
    end
    
    return
end



function update_vz!(vz,fact,σxz,σzz,dt,ρ_ihalf_jhalf,b_x_half,b_z_half,a_x_half,a_z_half,
                    freetop)

    if freetop
        for j = 1:2         
            for i = 2:nx-2     

                # Vz
                # σxz derivative only in x so no problem
                ∂σxz∂x_fwd = fact * ( σxz[i-1,j] -27.0*σxz[i,j] +27.0*σxz[i+1,j] -σxz[i+2,j] )
                # image, mirroring σzz[i,j-1] = -σxz[i,j+2], etc.
                ∂σzz∂z_fwd = fact * ( -σzz[i,j+2] +27.0*σzz[i,j+1] +27.0*σzz[i,j+1] -σzz[i,j+2] )
                # update velocity (ρ has been interpolated in advance)
                vz[i,j] = vz[i,j] + (dt/ρ_ihalf_jhalf[i,j]) * (∂σxz∂x_fwd + ∂σzz∂z_fwd)
                
            end
        end
    end

    #  vz       
    for j = 2:nz-2
        for i = 2:nx-2
            
            # Vz
            ∂σxz∂x_fwd = fact * ( σxz[i-1,j] -27.0*σxz[i,j] +27.0*σxz[i+1,j] -σxz[i+2,j] )
            ∂σzz∂z_fwd = fact * ( σzz[i,j-1] -27.0*σzz[i,j] +27.0*σzz[i,j+1] -σzz[i,j+2] )
            
            # C-PML stuff 
            ψ_∂σxz∂x[i,j] = b_x_half[i] * ψ_∂σxz∂x[i,j] + a_x_half[i]*∂σxz∂x_fwd
            ψ_∂σzz∂z[i,j] = b_z_half[j] * ψ_∂σzz∂z[i,j] + a_z_half[j]*∂σzz∂z_fwd
            
            ∂σxz∂x_fwd = ∂σxz∂x_fwd + ψ_∂σxz∂x[i,j]
            ∂σzz∂z_fwd = ∂σzz∂z_fwd + ψ_∂σzz∂z[i,j]

            # update velocity (ρ has been interpolated in advance)
            vz[i,j] = vz[i,j] + (dt/ρ_ihalf_jhalf[i,j]) * (∂σxz∂x_fwd + ∂σzz∂z_fwd)
            
        end
    end

    return
end


function update_σxxσzz!(σxx,σzz,fact,vx,λ_ihalf,μ_ihalf,dt,b_x_half,b_z,a_x_half,a_z,
                        freetop)

    if freetop==true
        # σxx, σzz
        # j=1: we are on the free surface!
        j = 1  
        for i = 2:nx-2                
            # σxx
            # vx derivative only in x so no problem
            ∂vx∂x_fwd = fact * ( vx[i-1,j] -27.0*vx[i,j] +27.0*vx[i+1,j] -vx[i+2,j] )
            # using boundary condition to calculate ∂vz∂z_bkd from ∂vx∂x_fwd
            ∂vz∂z_bkd = -(1.0-2.0*μ_ihalf[i,j]/λ_ihalf[i,j])*∂vx∂x_fwd
            # σxx
            σxx[i,j] = σxx[i,j] + (λ_ihalf[i,j]+2.0*μ_ihalf[i,j]) * dt * ∂vx∂x_fwd +
                λ_ihalf[i,j] * dt * ∂vz∂z_bkd
            
            # σzz
            σzz[i,j] = 0.0 # we are on the free surface!
        end
        
        # j=2: we are just below the surface (1/2)
        j = 2
        for i = 2:nx-2  
            # σxx
            # vx derivative only in x so no problem
            ∂vx∂x_fwd = fact * ( vx[i-1,j] -27.0*vx[i,j] +27.0*vx[i+1,j] -vx[i+2,j] )
            # zero velocity above the free surface
            ∂vz∂z_bkd = fact * ( 0.0 -27.0*vz[i,j-1] +27.0*vz[i,j] -vz[i,j+1] )
            # σxx
            σxx[i,j] = σxx[i,j] + (λ_ihalf[i,j]+2.0*μ_ihalf[i,j]) * dt * ∂vx∂x_fwd +
                λ_ihalf[i,j] * dt * ∂vz∂z_bkd
            
            # σzz
            σzz[i,j] = σzz[i,j] + (λ_ihalf[i,j]+2.0*μ_ihalf[i,j]) * dt* ∂vz∂z_bkd +
                λ_ihalf[i,j] * dt * ∂vx∂x_fwd

        end
    end

    #  σxx, σzz 
    for j = 3:nz-1
        for i = 2:nx-2                
            
            # σxx,σzz
            ∂vx∂x_fwd = fact * ( vx[i-1,j] -27.0*vx[i,j] +27.0*vx[i+1,j] -vx[i+2,j] )
            ∂vz∂z_bkd = fact * ( vz[i,j-2] -27.0*vz[i,j-1] +27.0*vz[i,j] -vz[i,j+1] )
            
            # C-PML stuff 
            ψ_∂vx∂x[i,j] = b_x_half[i] * ψ_∂vx∂x[i,j] + a_x_half[i]*∂vx∂x_fwd
            ψ_∂vz∂z[i,j] = b_z[j] * ψ_∂vz∂z[i,j] + a_z[j]*∂vz∂z_bkd

            ∂vx∂x_fwd = ∂vx∂x_fwd + ψ_∂vx∂x[i,j]
            ∂vz∂z_bkd = ∂vz∂z_bkd + ψ_∂vz∂z[i,j]
            
            # σxx
            σxx[i,j] = σxx[i,j] + (λ_ihalf[i,j]+2.0*μ_ihalf[i,j]) * dt * ∂vx∂x_fwd +
                λ_ihalf[i,j] * dt * ∂vz∂z_bkd

            ## derivatives are the same than for σxx 
            # σzz
            σzz[i,j] = σzz[i,j] + (λ_ihalf[i,j]+2.0*μ_ihalf[i,j]) * dt * ∂vz∂z_bkd +
                λ_ihalf[i,j] * dt * ∂vx∂x_fwd
            
        end
    end
    return
end


function update_σxz!(σxz,fact,vx,vz,μ_jhalf,dt,b_x,b_z_half,a_x,a_z_half,
                     freetop)
    
    if freetop
        # σxz
        j = 1
        for i=3:nx-1    
            # zero velocity above the free surface
            ∂vx∂z_fwd = fact * ( 0.0 -27.0*vx[i,j] +27.0*vx[i,j+1] -vx[i,j+2] )
            # vz derivative only in x so no problem
            ∂vz∂x_bkd = fact * ( vz[i-2,j] -27.0*vz[i-1,j] +27.0*vz[i,j] -vz[i+1,j] )
            
            # σxz
            σxz[i,j] = σxz[i,j] + μ_jhalf[i,j] * dt * (∂vx∂z_fwd + ∂vz∂x_bkd)

        end
    end

    #  σxz
    for j = 2:nz-2
        for i = 3:nx-1  

            # σxz
            ∂vx∂z_fwd = fact * ( vx[i,j-1] -27.0*vx[i,j] +27.0*vx[i,j+1] -vx[i,j+2] )
            ∂vz∂x_bkd = fact * ( vz[i-2,j] -27.0*vz[i-1,j] +27.0*vz[i,j] -vz[i+1,j] )

            # C-PML stuff 
            ψ_∂vx∂z[i,j] = b_z_half[j] * ψ_∂vx∂z[i,j] + a_z_half[j]*∂vx∂z_fwd
            ψ_∂vz∂x[i,j] = b_x[i] * ψ_∂vz∂x[i,j] + a_x[i]*∂vz∂x_bkd

            ∂vx∂z_fwd = ∂vx∂z_fwd + ψ_∂vx∂z[i,j]
            ∂vz∂x_bkd = ∂vz∂x_bkd + ψ_∂vz∂x[i,j]
            
            # σxz
            σxz[i,j] = σxz[i,j] + μ_jhalf[i,j] * dt * (∂vx∂z_fwd + ∂vz∂x_bkd)

        end
    end

    return
end



function forward_onestep_CPML!(vx,vz,σxx,σzz,σxz,
                               ρ,ρ_ihalf_jhalf,
                               μ,μ_ihalf,μ_jhalf,
                               dt,
                               b_x,b_z,b_x_half,b_z_half,
                               a_x,a_z,a_x_half,a_z_half,
                               freetop,save_trace)
                               

    # precompute some stuff
    μ_ihalf,μ_jhalf,λ_ihalf,fact = precomp_prop(ρ,μ,λ,dh)

    # update velocities vx and vz
    update_vx!(vx,fact,σxx,σxz,dt,ρ,ψ_∂σxx∂x,ψ_∂σxz∂z,b_x,b_z,a_x,a_z,
               freetop)

    update_vz!(vz,fact,σxz,σzz,dt,ρ_ihalf_jhalf,b_x_half,b_z_half,
               a_x_half,a_z_half,freetop)

    # update stresses σxx, σzz and σxz
    update_σxxσzz!(σxx,σzz,fact,vx,λ_ihalf,μ_ihalf,dt,b_x_half,
                   b_z,a_x_half,a_z,freetop)

    update_σxz!(σxz,fact,vx,vz,μ_jhalf,dt,b_x,b_z_half,
                a_x,a_z_half,freetop)
    

    # inject sources
    inject_sources!( ??????, dt2srctf, possrcs, it)  
    # record receivers
    if save_trace
        record_receivers!( ??????, traces, posrecs, it)
    end
    
    return
end


function correlate_gradient!(  )

    # _dt2 = 1 / dt^2
    # nx, ny = size(curgrad)
    # for j in 1:ny
    #     for i in 1:nx
    #         curgrad[i, j] = curgrad[i, j] + (adjcur[i, j] * (pcur[i, j] - 2.0 * pold[i, j] + pveryold[i, j]) * _dt2)
    #     end
    # end
    return 
end


#########################################
end  # end module
#########################################
