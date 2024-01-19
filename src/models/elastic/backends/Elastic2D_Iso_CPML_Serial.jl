
module Elastic2D_Iso_CPML_Serial


# To get those structs into this module
using SeismicWaves: ElasticIsoCPMLWaveSimul,ElasticIsoMaterialProperties

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



function update_4thord_vx!(nx,nz,halo,vx,factx,factz,σxx,σxz,dt,ρ,ψ_∂σxx∂x,ψ_∂σxz∂z,b_x,b_z,a_x,a_z,
                           freetop)
    if freetop
        for j = 1:2
            for i = 3:nx-1
                
                # Vx
                # σxx derivative only in x so no problem
                ∂σxx∂x_bkw = factx * ( σxx[i-2,j] -27.0*σxx[i-1,j] +27.0*σxx[i,j] -σxx[i+1,j] )
                # image, mirroring σxz[i,j-2] = -σxz[i,j+1], etc.
                #∂σxz∂z_bkw = factz * ( -σxz[i,j+1] +27.0*σxz[i,j] +27.0*σxz[i,j] -σxz[i,j+1] )
                if j==1
                    # j bwd-> -2 -1|0 1 (mirror -2 and -1)
                    ∂σxz∂z_bkw = factz * ( -σxz[i,j+1] +27.0*σxz[i,j] +27.0*σxz[i,j] -σxz[i,j+1] )
                elseif j==2
                    # j bwd-> -2|-1 0 1 (mirror only -2)
                    ∂σxz∂z_bkw = factz * ( -σxz[i,j] -27.0*σxz[i,j-1] +27.0*σxz[i,j] -σxz[i,j+1] )
                end
                # update velocity
                vx[i,j] = vx[i,j] + (dt/ρ[i,j]) * (∂σxx∂x_bkw + ∂σxz∂z_bkw)

            end
        end
    end
    
    #  vx
    for j = 3:nz-1
        for i = 3:nx-1
            
            # Vx
            ∂σxx∂x_bkw = factx * ( σxx[i-2,j] -27.0*σxx[i-1,j] +27.0*σxx[i,j] -σxx[i+1,j] )
            ∂σxz∂z_bkw = factz * ( σxz[i,j-2] -27.0*σxz[i,j-1] +27.0*σxz[i,j] -σxz[i,j+1] )

            ##=======================
            # C-PML stuff
            ##=======================
            # x boundaries
            
            # con1 = (i <= halo)
            # con2 = (i >= nx - halo + 1)
            # if con1 || con2
            #     ii = i
            #     if con2
            #         ii = i - (nx - 2*halo) + 1
            #     end
            #     # left or right boundary
            #     ψ_∂σxx∂x[ii,j] = b_x[ii] * ψ_∂σxx∂x[ii,j] + a_x[ii] * ∂σxx∂x_bkw
            #     ∂σxx∂x_bkw = ∂σxx∂x_bkw + ψ_∂σxx∂x[ii,j]
            # end

            if i <= halo  # 
                # left boundary
                ψ_∂σxx∂x[i,j] = b_x[i] * ψ_∂σxx∂x[i,j] + a_x[i] * ∂σxx∂x_bkw
                ∂σxx∂x_bkw = ∂σxx∂x_bkw + ψ_∂σxx∂x[i,j]
            elseif i >= nx - halo + 1 
                # right boundary
                ii = i - (nx - 2*halo) 
                ψ_∂σxx∂x[ii,j] = b_x[ii] * ψ_∂σxx∂x[ii,j] + a_x[ii] * ∂σxx∂x_bkw
                ∂σxx∂x_bkw = ∂σxx∂x_bkw + ψ_∂σxx∂x[ii,j]
            end
            # y boundaries
            if j <= halo && freetop==false 
                # top boundary
                ψ_∂σxz∂z[i,j] = b_z[j] * ψ_∂σxz∂z[i,j] + a_z[j] * ∂σxz∂z_bkw
                ∂σxz∂z_bkw = ∂σxz∂z_bkw + ψ_∂σxz∂z[i,j]
            elseif j >= nz - halo + 1 
                # bottom boundary
                jj = j - (nz - 2*halo) 
                ψ_∂σxz∂z[i,jj] = b_z[jj] * ψ_∂σxz∂z[i,jj] + a_z[jj] * ∂σxz∂z_bkw
                ∂σxz∂z_bkw = ∂σxz∂z_bkw + ψ_∂σxz∂z[i,jj]
            end
            ##=======================
            
            # # C-PML stuff
            # # DO NOT delete this part!
            # ψ_∂σxx∂x[i,j] = b_x[i] * ψ_∂σxx∂x[i,j] + a_x[i] * ∂σxx∂x_bkw
            # ψ_∂σxz∂z[i,j] = b_z[j] * ψ_∂σxz∂z[i,j] + a_z[j] * ∂σxz∂z_bkw
            # # derivatives
            # ∂σxx∂x_bkw = ∂σxx∂x_bkw + ψ_∂σxx∂x[i,j]
            # ∂σxz∂z_bkw = ∂σxz∂z_bkw + ψ_∂σxz∂z[i,j]

            # update velocity
            vx[i,j] = vx[i,j] + (dt/ρ[i,j]) * (∂σxx∂x_bkw + ∂σxz∂z_bkw)

        end
    end    
    return
end



function update_4thord_vz!(nx,nz,halo,vz,factx,factz,σxz,σzz,dt,ρ_ihalf_jhalf,ψ_∂σxz∂x,ψ_∂σzz∂z,
                           b_x_half,b_z_half,a_x_half,a_z_half,freetop)

    if freetop
        for j = 1:2         
            for i = 2:nx-2     

                # Vz
                # σxz derivative only in x so no problem
                ∂σxz∂x_fwd = factx * ( σxz[i-1,j] -27.0*σxz[i,j] +27.0*σxz[i+1,j] -σxz[i+2,j] )
                # image, mirroring σzz[i,j-1] = -σxz[i,j+2], etc.
                #∂σzz∂z_fwd = factz * ( -σzz[i,j+2] +27.0*σzz[i,j+1] +27.0*σzz[i,j+1] -σzz[i,j+2] )
                if j==1
                    # j fwd-> -1 0| 1 2 (mirror -2 and -1)
                    ∂σzz∂z_fwd = factz * ( -σzz[i,j+2] +27.0*σzz[i,j+1] +27.0*σzz[i,j+1] -σzz[i,j+2] )
                elseif j==2
                    # j fwd-> -1|0 1 2 (mirror only -1)
                    ∂σzz∂z_fwd = factz * ( -σzz[i,j+2] -27.0*σzz[i,j] +27.0*σzz[i,j+1] -σzz[i,j+2] )
                end
                # update velocity (ρ has been interpolated in advance)
                vz[i,j] = vz[i,j] + (dt/ρ_ihalf_jhalf[i,j]) * (∂σxz∂x_fwd + ∂σzz∂z_fwd)
                
            end
        end
    end

    #  vz       
    for j = 2:nz-2
        for i = 2:nx-2
            
            # Vz
            ∂σxz∂x_fwd = factx * ( σxz[i-1,j] -27.0*σxz[i,j] +27.0*σxz[i+1,j] -σxz[i+2,j] )
            ∂σzz∂z_fwd = factz * ( σzz[i,j-1] -27.0*σzz[i,j] +27.0*σzz[i,j+1] -σzz[i,j+2] )


            ##=======================
            # C-PML stuff
            ##=======================
            # x boundaries
            if i <= halo 
                # left boundary
                ψ_∂σxz∂x[i,j] = b_x_half[i] * ψ_∂σxz∂x[i,j] + a_x_half[i]*∂σxz∂x_fwd
                ∂σxz∂x_fwd = ∂σxz∂x_fwd + ψ_∂σxz∂x[i,j]
            elseif i >= nx - halo + 1
                # right boundary
                ii = i - (nx - 2*halo) 
                ψ_∂σxz∂x[ii,j] = b_x_half[ii] * ψ_∂σxz∂x[ii,j] + a_x_half[ii]*∂σxz∂x_fwd
                ∂σxz∂x_fwd = ∂σxz∂x_fwd + ψ_∂σxz∂x[ii,j]
            end
            # y boundaries
            if j <= halo && freetop==false # + 1
                # top boundary
                ψ_∂σzz∂z[i,j] = b_z_half[j] * ψ_∂σzz∂z[i,j] + a_z_half[j]*∂σzz∂z_fwd
                ∂σzz∂z_fwd = ∂σzz∂z_fwd + ψ_∂σzz∂z[i,j]
            elseif j >= nz - halo + 1
                # bottom boundary
                jj = j - (nz - 2*halo) 
                ψ_∂σzz∂z[i,jj] = b_z_half[jj] * ψ_∂σzz∂z[i,jj] + a_z_half[jj]*∂σzz∂z_fwd
                ∂σzz∂z_fwd = ∂σzz∂z_fwd + ψ_∂σzz∂z[i,jj]
            end
            ##=======================

            # # C-PML stuff
            # # DO NOT delete this part!
            # ψ_∂σxz∂x[i,j] = b_x_half[i] * ψ_∂σxz∂x[i,j] + a_x_half[i]*∂σxz∂x_fwd
            # ψ_∂σzz∂z[i,j] = b_z_half[j] * ψ_∂σzz∂z[i,j] + a_z_half[j]*∂σzz∂z_fwd
            # ∂σxz∂x_fwd = ∂σxz∂x_fwd + ψ_∂σxz∂x[i,j]
            # ∂σzz∂z_fwd = ∂σzz∂z_fwd + ψ_∂σzz∂z[i,j]

            # update velocity (ρ has been interpolated in advance)
            vz[i,j] = vz[i,j] + (dt/ρ_ihalf_jhalf[i,j]) * (∂σxz∂x_fwd + ∂σzz∂z_fwd)
            
        end
    end
    return
end


function update_4thord_σxxσzz!(nx,nz,halo,σxx,σzz,factx,factz,
                               vx,vz,dt,λ_ihalf,μ_ihalf,ψ_∂vx∂x,ψ_∂vz∂z,
                               b_x_half,b_z,a_x_half,a_z,freetop)

    if freetop==true
        # σxx, σzz
        # j=1: we are on the free surface!
        j = 1  
        for i = 2:nx-2                
            # σxx
            # vx derivative only in x so no problem
            ∂vx∂x_fwd = factx * ( vx[i-1,j] -27.0*vx[i,j] +27.0*vx[i+1,j] -vx[i+2,j] )
            # using boundary condition to calculate ∂vz∂z_bkd from ∂vx∂x_fwd
            ∂vz∂z_bkd = -(1.0-2.0*μ_ihalf[i,j]/λ_ihalf[i,j])*∂vx∂x_fwd
            # σxx
            # σxx[i,j] = σxx[i,j] + (λ_ihalf[i,j]+2.0*μ_ihalf[i,j]) * dt * ∂vx∂x_fwd + λ_ihalf[i,j] * dt * ∂vz∂z_bkd
            σxx[i,j] = σxx[i,j] + (λ_ihalf[i,j] - λ_ihalf[i,j] / (λ_ihalf[i,j]+2+μ_ihalf[i,j]) + 2*μ_ihalf[i,j]) * dt * ∂vx∂x_fwd
            # σzz
            σzz[i,j] = 0.0 # we are on the free surface!
        end

        # j=2: we are just below the surface (1/2)
        j = 2
        for i = 2:nx-2  
            # σxx
            # vx derivative only in x so no problem
            ∂vx∂x_fwd = factx * ( vx[i-1,j] -27.0*vx[i,j] +27.0*vx[i+1,j] -vx[i+2,j] )
            # zero velocity above the free surface
            ∂vz∂z_bkd = factz * ( 0.0 -27.0*vz[i,j-1] +27.0*vz[i,j] -vz[i,j+1] )
            # σxx
            σxx[i,j] = σxx[i,j] + (λ_ihalf[i,j]+2.0*μ_ihalf[i,j]) * dt * ∂vx∂x_fwd +
                λ_ihalf[i,j] * dt * ∂vz∂z_bkd            
            # σzz
            σzz[i,j] = σzz[i,j] + (λ_ihalf[i,j]+2.0*μ_ihalf[i,j]) * dt * ∂vz∂z_bkd +
                λ_ihalf[i,j] * dt * ∂vx∂x_fwd
        end
    end

    #  σxx, σzz 
    for j = 3:nz-1
        for i = 2:nx-2                
            
            # σxx,σzz
            ∂vx∂x_fwd = factx * ( vx[i-1,j] -27.0*vx[i,j] +27.0*vx[i+1,j] -vx[i+2,j] )
            ∂vz∂z_bkd = factz * ( vz[i,j-2] -27.0*vz[i,j-1] +27.0*vz[i,j] -vz[i,j+1] )

            ##=======================
            # C-PML stuff
            ##=======================
            # x boundaries
            if i <= halo 
                # left boundary
                ψ_∂vx∂x[i,j] = b_x_half[i] * ψ_∂vx∂x[i,j] + a_x_half[i]*∂vx∂x_fwd
                ∂vx∂x_fwd = ∂vx∂x_fwd + ψ_∂vx∂x[i,j]
            elseif i >= nx - halo + 1
                # right boundary
                ii = i - (nx - 2*halo) 
                ψ_∂vx∂x[ii,j] = b_x_half[ii] * ψ_∂vx∂x[ii,j] + a_x_half[ii]*∂vx∂x_fwd
                ∂vx∂x_fwd = ∂vx∂x_fwd + ψ_∂vx∂x[ii,j]
            end
            # y boundaries
            if j <= halo && freetop==false 
                # top boundary
                ψ_∂vz∂z[i,j] = b_z[j] * ψ_∂vz∂z[i,j] + a_z[j]*∂vz∂z_bkd
                ∂vz∂z_bkd = ∂vz∂z_bkd + ψ_∂vz∂z[i,j]
            elseif j >= nz - halo + 1
                # bottom boundary
                jj = j - (nz - 2*halo)
                ψ_∂vz∂z[i,jj] = b_z[jj] * ψ_∂vz∂z[i,jj] + a_z[jj]*∂vz∂z_bkd
                ∂vz∂z_bkd = ∂vz∂z_bkd + ψ_∂vz∂z[i,jj]
            end
            ##=======================
            
            # # C-PML stuff
            # # DO NOT delete this part!
            # ψ_∂vx∂x[i,j] = b_x_half[i] * ψ_∂vx∂x[i,j] + a_x_half[i]*∂vx∂x_fwd
            # ψ_∂vz∂z[i,j] = b_z[j] * ψ_∂vz∂z[i,j] + a_z[j]*∂vz∂z_bkd
            # ∂vx∂x_fwd = ∂vx∂x_fwd + ψ_∂vx∂x[i,j]
            # ∂vz∂z_bkd = ∂vz∂z_bkd + ψ_∂vz∂z[i,j]
            
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


function update_4thord_σxz!(nx,nz,halo,σxz,factx,factz,vx,vz,dt,
                     μ_jhalf,b_x,b_z_half,
                     ψ_∂vx∂z,ψ_∂vz∂x,a_x,a_z_half,
                     freetop)
    
    if freetop
        # σxz
        j = 1
        for i=3:nx-1    
            # zero velocity above the free surface
            ∂vx∂z_fwd = factz * ( 0.0 -27.0*vx[i,j] +27.0*vx[i,j+1] -vx[i,j+2] )
            # vz derivative only in x so no problem
            ∂vz∂x_bkd = factx * ( vz[i-2,j] -27.0*vz[i-1,j] +27.0*vz[i,j] -vz[i+1,j] )
            # σxz
            σxz[i,j] = σxz[i,j] + μ_jhalf[i,j] * dt * (∂vx∂z_fwd + ∂vz∂x_bkd)
        end
    end

    #  σxz
    for j = 2:nz-2
        for i = 3:nx-1  

            # σxz
            ∂vx∂z_fwd = factz * ( vx[i,j-1] -27.0*vx[i,j] +27.0*vx[i,j+1] -vx[i,j+2] )
            ∂vz∂x_bkd = factx * ( vz[i-2,j] -27.0*vz[i-1,j] +27.0*vz[i,j] -vz[i+1,j] )
            
            ##=======================
            # C-PML stuff
            ##=======================
            # x boundaries
            if i <= halo 
                # left boundary
                ψ_∂vz∂x[i,j] = b_x[i] * ψ_∂vz∂x[i,j] + a_x[i]*∂vz∂x_bkd
                ∂vz∂x_bkd = ∂vz∂x_bkd + ψ_∂vz∂x[i,j]
            elseif i >= nx - halo + 1
                # right boundary
                ii = i - (nx - 2*halo) 
                ψ_∂vz∂x[ii,j] = b_x[ii] * ψ_∂vz∂x[ii,j] + a_x[ii]*∂vz∂x_bkd
                ∂vz∂x_bkd = ∂vz∂x_bkd + ψ_∂vz∂x[ii,j]
            end
            # y boundaries
            if j <= halo  && freetop==false 
                # top boundary
                ψ_∂vx∂z[i,j] = b_z_half[j] * ψ_∂vx∂z[i,j] + a_z_half[j]*∂vx∂z_fwd
                ∂vx∂z_fwd = ∂vx∂z_fwd + ψ_∂vx∂z[i,j]
            elseif j >= nz - halo + 1
                # bottom boundary
                jj = j - (nz - 2*halo) 
                ψ_∂vx∂z[i,jj] = b_z_half[jj] * ψ_∂vx∂z[i,jj] + a_z_half[jj]*∂vx∂z_fwd
                ∂vx∂z_fwd = ∂vx∂z_fwd + ψ_∂vx∂z[i,jj]
            end
            ##=======================
            
            # # C-PML stuff
            # # DO NOT delete this part!
            # ψ_∂vz∂x[i,j] = b_x[i] * ψ_∂vz∂x[i,j] + a_x[i]*∂vz∂x_bkd
            # ψ_∂vx∂z[i,j] = b_z_half[j] * ψ_∂vx∂z[i,j] + a_z_half[j]*∂vx∂z_fwd
            # ∂vz∂x_bkd = ∂vz∂x_bkd + ψ_∂vz∂x[i,j]
            # ∂vx∂z_fwd = ∂vx∂z_fwd + ψ_∂vx∂z[i,j]
            
            # σxz
            σxz[i,j] = σxz[i,j] + μ_jhalf[i,j] * dt * (∂vx∂z_fwd + ∂vz∂x_bkd)

        end
    end
    return
end



function forward_onestep_CPML!(wavsim::ElasticIsoCPMLWaveSimul{N},
                               possrcs_bk::Array{<:Integer,2},
                               srctf_bk::Matrix{<:Real},
                               posrecs_bk::Array{<:Integer,2},
                               traces_bk::Array{<:Real},
                               it::Integer,
                               Mxx::Vector{<:Real},
                               Mzz::Vector{<:Real},
                               Mxz::Vector{<:Real};
                               save_trace::Bool=true) where {N}

    @assert N==2
    freetop = wavsim.freetop
    cpmlcoeffs = wavsim.cpmlcoeffs
    matprop = wavsim.matprop
    dx = wavsim.gridspacing[1]
    dz = wavsim.gridspacing[2]
    dt = wavsim.dt
    nx,nz = wavsim.gridsize[1:2]
    halo = wavsim.halo

    vx = wavsim.velpartic.vx
    vz = wavsim.velpartic.vz
    σxx = wavsim.stress.σxx
    σzz = wavsim.stress.σzz
    σxz = wavsim.stress.σxz
    
    psi = wavsim.ψ

    a_x = cpmlcoeffs[1].a
    a_x_half = cpmlcoeffs[1].a_h
    b_x = cpmlcoeffs[1].b
    b_x_half = cpmlcoeffs[1].b_h

    a_z = cpmlcoeffs[2].a
    a_z_half = cpmlcoeffs[2].a_h
    b_z = cpmlcoeffs[2].b
    b_z_half = cpmlcoeffs[2].b_h

    λ_ihalf = matprop.λ_ihalf
    ρ = matprop.ρ
    ρ_ihalf_jhalf = matprop.ρ_ihalf_jhalf
    μ = matprop.μ
    μ_ihalf = matprop.μ_ihalf
    μ_jhalf = matprop.μ_jhalf

    ## pre-scale coefficients
    factx = 1.0/(24.0*dx)
    factz = 1.0/(24.0*dz)
    
    # update velocity vx 
    update_4thord_vx!(nx,nz,halo,vx,factx,factz,σxx,σxz,dt,ρ,psi.ψ_∂σxx∂x,psi.ψ_∂σxz∂z,
                      b_x,b_z,a_x,a_z,freetop)    
    # update velocity vz
    update_4thord_vz!(nx,nz,halo,vz,factx,factz,σxz,σzz,dt,ρ_ihalf_jhalf,psi.ψ_∂σxz∂x,
                      psi.ψ_∂σzz∂z,b_x_half,b_z_half,a_x_half,a_z_half,freetop)

    # inject sources (external body force)
    #inject_bodyforce_sources!(vx,vz,fx,fz,srctf_bk, dt, possrcs_bk,it)

    # update stresses σxx and σzz 
    update_4thord_σxxσzz!(nx,nz,halo,σxx,σzz,factx,factz,
                          vx,vz,dt,λ_ihalf,μ_ihalf,
                          psi.ψ_∂vx∂x,psi.ψ_∂vz∂z,
                          b_x_half,b_z,a_x_half,a_z,freetop)
    # update stress σxz
    update_4thord_σxz!(nx,nz,halo,σxz,factx,factz,vx,vz,dt,
                       μ_jhalf,b_x,b_z_half,
                       psi.ψ_∂vx∂z,psi.ψ_∂vz∂x,a_x,a_z_half,freetop)

    # inject sources (moment tensor type of internal force)
    inject_momten_sources!(σxx,σzz,σxz,Mxx,Mzz,Mxz, srctf_bk, dt, possrcs_bk, it)
    
    # record receivers
    if save_trace
        record_receivers2D!(vx,vz,traces_bk, posrecs_bk, it)
    end
    
    return
end


function inject_momten_sources!(σxx,σzz,σxz,Mxx, Mzz, Mxz, srctf_bk, dt, possrcs_bk, it)

    ## Inject the source as stress from moment tensor
    ##  See Igel 2017 Computational Seismology (book) page 31, 2.6.1
    lensrctf = length(srctf_bk)
    if it<=lensrctf

        for s in axes(possrcs_bk, 1)
            isrc = possrcs_bk[s, 1]
            jsrc = possrcs_bk[s, 2]

            σxx[isrc,jsrc] += Mxx[s] * srctf_bk[it] * dt 
            σzz[isrc,jsrc] += Mzz[s] * srctf_bk[it] * dt 
            σxz[isrc,jsrc] += Mxz[s] * srctf_bk[it] * dt
        end
    end    
    return
end


function record_receivers2D!(vx,vz,traces_bk, posrecs, it)

    for ir in axes(posrecs, 1)
        irec = posrecs[ir, 1]
        jrec = posrecs[ir, 2]
        # interpolate velocities on the same grid?
        traces_bk[it,1,ir] = vx[irec, jrec]
        traces_bk[it,2,ir] = vz[irec, jrec]
    end
    return
end


#function correlate_gradient!(  )
    # _dt2 = 1 / dt^2
    # nx, nz = size(curgrad)
    # for j in 1:nz
    #     for i in 1:nx
    #         curgrad[i, j] = curgrad[i, j] + (adjcur[i, j] * (pcur[i, j] - 2.0 * pold[i, j] + pveryold[i, j]) * _dt2)
    #     end
    # end
    # return 
#end


#########################################
end  # end module
#########################################
