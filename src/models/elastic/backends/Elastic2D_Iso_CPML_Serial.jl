
module Elastic2D_Iso_CPML_Serial

# To get those structs into this module
using SeismicWaves: ElasticIsoCPMLWaveSimulation, ElasticIsoMaterialProperties

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

function update_4thord_vx!(nx, nz, halo, vx, factx, factz, σxx, σxz, dt, ρ, ψ_∂σxx∂x, ψ_∂σxz∂z, b_x, b_z, a_x, a_z,
    freetop)
    if freetop
        for j in 1:2
            for i in 3:nx-1

                # Vx
                # σxx derivative only in x so no problem
                ∂σxx∂x_bkw = factx * (σxx[i-2, j] - 27.0 * σxx[i-1, j] + 27.0 * σxx[i, j] - σxx[i+1, j])
                # image, mirroring σxz[i,j-2] = -σxz[i,j+1], etc.
                #∂σxz∂z_bkw = factz * ( -σxz[i,j+1] +27.0*σxz[i,j] +27.0*σxz[i,j] -σxz[i,j+1] )
                if j == 1
                    # j bwd-> -2 -1|0 1 (mirror -2 and -1)
                    ∂σxz∂z_bkw = factz * (-σxz[i, j+1] + 27.0 * σxz[i, j] + 27.0 * σxz[i, j] - σxz[i, j+1])
                elseif j == 2
                    # j bwd-> -2|-1 0 1 (mirror only -2)
                    ∂σxz∂z_bkw = factz * (-σxz[i, j] - 27.0 * σxz[i, j-1] + 27.0 * σxz[i, j] - σxz[i, j+1])
                end
                # update velocity
                vx[i, j] = vx[i, j] + (dt / ρ[i, j]) * (∂σxx∂x_bkw + ∂σxz∂z_bkw)
            end
        end
    end

    #  vx
    for j in 3:nz-1
        for i in 3:nx-1

            # Vx
            ∂σxx∂x_bkw = factx * (σxx[i-2, j] - 27.0 * σxx[i-1, j] + 27.0 * σxx[i, j] - σxx[i+1, j])
            ∂σxz∂z_bkw = factz * (σxz[i, j-2] - 27.0 * σxz[i, j-1] + 27.0 * σxz[i, j] - σxz[i, j+1])

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
                ψ_∂σxx∂x[i, j] = b_x[i] * ψ_∂σxx∂x[i, j] + a_x[i] * ∂σxx∂x_bkw
                ∂σxx∂x_bkw = ∂σxx∂x_bkw + ψ_∂σxx∂x[i, j]
            elseif i >= nx - halo + 1
                # right boundary
                ii = i - (nx - 2 * halo)
                ψ_∂σxx∂x[ii, j] = b_x[ii] * ψ_∂σxx∂x[ii, j] + a_x[ii] * ∂σxx∂x_bkw
                ∂σxx∂x_bkw = ∂σxx∂x_bkw + ψ_∂σxx∂x[ii, j]
            end
            # y boundaries
            if j <= halo && freetop == false
                # top boundary
                ψ_∂σxz∂z[i, j] = b_z[j] * ψ_∂σxz∂z[i, j] + a_z[j] * ∂σxz∂z_bkw
                ∂σxz∂z_bkw = ∂σxz∂z_bkw + ψ_∂σxz∂z[i, j]
            elseif j >= nz - halo + 1
                # bottom boundary
                jj = j - (nz - 2 * halo)
                ψ_∂σxz∂z[i, jj] = b_z[jj] * ψ_∂σxz∂z[i, jj] + a_z[jj] * ∂σxz∂z_bkw
                ∂σxz∂z_bkw = ∂σxz∂z_bkw + ψ_∂σxz∂z[i, jj]
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
            vx[i, j] = vx[i, j] + (dt / ρ[i, j]) * (∂σxx∂x_bkw + ∂σxz∂z_bkw)
        end
    end
    return
end

function update_4thord_vz!(nx, nz, halo, vz, factx, factz, σxz, σzz, dt, ρ_ihalf_jhalf, ψ_∂σxz∂x, ψ_∂σzz∂z,
    b_x_half, b_z_half, a_x_half, a_z_half, freetop)
    if freetop
        for j in 1:2
            for i in 2:nx-2

                # Vz
                # σxz derivative only in x so no problem
                ∂σxz∂x_fwd = factx * (σxz[i-1, j] - 27.0 * σxz[i, j] + 27.0 * σxz[i+1, j] - σxz[i+2, j])
                # image, mirroring σzz[i,j-1] = -σxz[i,j+2], etc.
                #∂σzz∂z_fwd = factz * ( -σzz[i,j+2] +27.0*σzz[i,j+1] +27.0*σzz[i,j+1] -σzz[i,j+2] )
                if j == 1
                    # j fwd-> -1 0| 1 2 (mirror -2 and -1)
                    ∂σzz∂z_fwd = factz * (-σzz[i, j+2] + 27.0 * σzz[i, j+1] + 27.0 * σzz[i, j+1] - σzz[i, j+2])
                elseif j == 2
                    # j fwd-> -1|0 1 2 (mirror only -1)
                    ∂σzz∂z_fwd = factz * (-σzz[i, j+2] - 27.0 * σzz[i, j] + 27.0 * σzz[i, j+1] - σzz[i, j+2])
                end
                # update velocity (ρ has been interpolated in advance)
                vz[i, j] = vz[i, j] + (dt / ρ_ihalf_jhalf[i, j]) * (∂σxz∂x_fwd + ∂σzz∂z_fwd)
            end
        end
    end

    #  vz       
    for j in 2:nz-2 # TODO maybe typo (2 -> 3)
        for i in 2:nx-2

            # Vz
            ∂σxz∂x_fwd = factx * (σxz[i-1, j] - 27.0 * σxz[i, j] + 27.0 * σxz[i+1, j] - σxz[i+2, j])
            ∂σzz∂z_fwd = factz * (σzz[i, j-1] - 27.0 * σzz[i, j] + 27.0 * σzz[i, j+1] - σzz[i, j+2])

            ##=======================
            # C-PML stuff
            ##=======================
            # x boundaries
            if i <= halo
                # left boundary
                ψ_∂σxz∂x[i, j] = b_x_half[i] * ψ_∂σxz∂x[i, j] + a_x_half[i] * ∂σxz∂x_fwd
                ∂σxz∂x_fwd = ∂σxz∂x_fwd + ψ_∂σxz∂x[i, j]
            elseif i >= nx - halo + 1
                # right boundary
                ii = i - (nx - 2 * halo)
                ψ_∂σxz∂x[ii, j] = b_x_half[ii] * ψ_∂σxz∂x[ii, j] + a_x_half[ii] * ∂σxz∂x_fwd
                ∂σxz∂x_fwd = ∂σxz∂x_fwd + ψ_∂σxz∂x[ii, j]
            end
            # y boundaries
            if j <= halo && freetop == false # + 1
                # top boundary
                ψ_∂σzz∂z[i, j] = b_z_half[j] * ψ_∂σzz∂z[i, j] + a_z_half[j] * ∂σzz∂z_fwd
                ∂σzz∂z_fwd = ∂σzz∂z_fwd + ψ_∂σzz∂z[i, j]
            elseif j >= nz - halo + 1
                # bottom boundary
                jj = j - (nz - 2 * halo)
                ψ_∂σzz∂z[i, jj] = b_z_half[jj] * ψ_∂σzz∂z[i, jj] + a_z_half[jj] * ∂σzz∂z_fwd
                ∂σzz∂z_fwd = ∂σzz∂z_fwd + ψ_∂σzz∂z[i, jj]
            end
            ##=======================

            # # C-PML stuff
            # # DO NOT delete this part!
            # ψ_∂σxz∂x[i,j] = b_x_half[i] * ψ_∂σxz∂x[i,j] + a_x_half[i]*∂σxz∂x_fwd
            # ψ_∂σzz∂z[i,j] = b_z_half[j] * ψ_∂σzz∂z[i,j] + a_z_half[j]*∂σzz∂z_fwd
            # ∂σxz∂x_fwd = ∂σxz∂x_fwd + ψ_∂σxz∂x[i,j]
            # ∂σzz∂z_fwd = ∂σzz∂z_fwd + ψ_∂σzz∂z[i,j]

            # update velocity (ρ has been interpolated in advance)
            vz[i, j] = vz[i, j] + (dt / ρ_ihalf_jhalf[i, j]) * (∂σxz∂x_fwd + ∂σzz∂z_fwd)
        end
    end
    return
end

function update_4thord_σxxσzz!(nx, nz, halo, σxx, σzz, factx, factz,
    vx, vz, dt, λ_ihalf, μ_ihalf, ψ_∂vx∂x, ψ_∂vz∂z,
    b_x_half, b_z, a_x_half, a_z, freetop)
    if freetop == true
        # σxx, σzz
        # j=1: we are on the free surface!
        j = 1
        for i in 2:nx-2
            # σxx
            # vx derivative only in x so no problem
            ∂vx∂x_fwd = factx * (vx[i-1, j] - 27.0 * vx[i, j] + 27.0 * vx[i+1, j] - vx[i+2, j])
            # using boundary condition to calculate ∂vz∂z_bkd from ∂vx∂x_fwd
            ∂vz∂z_bkd = -(λ_ihalf[i, j] / (λ_ihalf[i, j] + 2.0 * μ_ihalf[i, j])) * ∂vx∂x_fwd
            # σxx
            σxx[i, j] = σxx[i, j] + (λ_ihalf[i, j] + 2.0 * μ_ihalf[i, j]) * dt * ∂vx∂x_fwd +
                        λ_ihalf[i, j] * dt * ∂vz∂z_bkd
            # σzz
            σzz[i, j] = 0.0 # we are on the free surface!
        end

        # j=2: we are just below the surface (1/2)
        j = 2
        for i in 2:nx-2
            # σxx
            # vx derivative only in x so no problem
            ∂vx∂x_fwd = factx * (vx[i-1, j] - 27.0 * vx[i, j] + 27.0 * vx[i+1, j] - vx[i+2, j])
            # zero velocity above the free surface
            ∂vz∂z_bkd = factz * (0.0 - 27.0 * vz[i, j-1] + 27.0 * vz[i, j] - vz[i, j+1])
            # σxx
            σxx[i, j] = σxx[i, j] + (λ_ihalf[i, j] + 2.0 * μ_ihalf[i, j]) * dt * ∂vx∂x_fwd +
                        λ_ihalf[i, j] * dt * ∂vz∂z_bkd
            # σzz
            σzz[i, j] = σzz[i, j] + (λ_ihalf[i, j] + 2.0 * μ_ihalf[i, j]) * dt * ∂vz∂z_bkd +
                        λ_ihalf[i, j] * dt * ∂vx∂x_fwd
        end
    end

    #  σxx, σzz 
    for j in 3:nz-1
        for i in 2:nx-2

            # σxx,σzz
            ∂vx∂x_fwd = factx * (vx[i-1, j] - 27.0 * vx[i, j] + 27.0 * vx[i+1, j] - vx[i+2, j])
            ∂vz∂z_bkd = factz * (vz[i, j-2] - 27.0 * vz[i, j-1] + 27.0 * vz[i, j] - vz[i, j+1])

            ##=======================
            # C-PML stuff
            ##=======================
            # x boundaries
            if i <= halo
                # left boundary
                ψ_∂vx∂x[i, j] = b_x_half[i] * ψ_∂vx∂x[i, j] + a_x_half[i] * ∂vx∂x_fwd
                ∂vx∂x_fwd = ∂vx∂x_fwd + ψ_∂vx∂x[i, j]
            elseif i >= nx - halo + 1
                # right boundary
                ii = i - (nx - 2 * halo)
                ψ_∂vx∂x[ii, j] = b_x_half[ii] * ψ_∂vx∂x[ii, j] + a_x_half[ii] * ∂vx∂x_fwd
                ∂vx∂x_fwd = ∂vx∂x_fwd + ψ_∂vx∂x[ii, j]
            end
            # y boundaries
            if j <= halo && freetop == false
                # top boundary
                ψ_∂vz∂z[i, j] = b_z[j] * ψ_∂vz∂z[i, j] + a_z[j] * ∂vz∂z_bkd
                ∂vz∂z_bkd = ∂vz∂z_bkd + ψ_∂vz∂z[i, j]
            elseif j >= nz - halo + 1
                # bottom boundary
                jj = j - (nz - 2 * halo)
                ψ_∂vz∂z[i, jj] = b_z[jj] * ψ_∂vz∂z[i, jj] + a_z[jj] * ∂vz∂z_bkd
                ∂vz∂z_bkd = ∂vz∂z_bkd + ψ_∂vz∂z[i, jj]
            end
            ##=======================

            # # C-PML stuff
            # # DO NOT delete this part!
            # ψ_∂vx∂x[i,j] = b_x_half[i] * ψ_∂vx∂x[i,j] + a_x_half[i]*∂vx∂x_fwd
            # ψ_∂vz∂z[i,j] = b_z[j] * ψ_∂vz∂z[i,j] + a_z[j]*∂vz∂z_bkd
            # ∂vx∂x_fwd = ∂vx∂x_fwd + ψ_∂vx∂x[i,j]
            # ∂vz∂z_bkd = ∂vz∂z_bkd + ψ_∂vz∂z[i,j]

            # σxx
            σxx[i, j] = σxx[i, j] + (λ_ihalf[i, j] + 2.0 * μ_ihalf[i, j]) * dt * ∂vx∂x_fwd +
                        λ_ihalf[i, j] * dt * ∂vz∂z_bkd

            ## derivatives are the same than for σxx 
            # σzz
            σzz[i, j] = σzz[i, j] + (λ_ihalf[i, j] + 2.0 * μ_ihalf[i, j]) * dt * ∂vz∂z_bkd +
                        λ_ihalf[i, j] * dt * ∂vx∂x_fwd
        end
    end
    return
end

function update_4thord_σxz!(nx, nz, halo, σxz, factx, factz, vx, vz, dt,
    μ_jhalf, b_x, b_z_half,
    ψ_∂vx∂z, ψ_∂vz∂x, a_x, a_z_half,
    freetop)
    if freetop
        # σxz
        j = 1
        for i in 3:nx-1
            # zero velocity above the free surface
            ∂vx∂z_fwd = factz * (0.0 - 27.0 * vx[i, j] + 27.0 * vx[i, j+1] - vx[i, j+2])
            # vz derivative only in x so no problem
            ∂vz∂x_bkd = factx * (vz[i-2, j] - 27.0 * vz[i-1, j] + 27.0 * vz[i, j] - vz[i+1, j])
            # σxz
            σxz[i, j] = σxz[i, j] + μ_jhalf[i, j] * dt * (∂vx∂z_fwd + ∂vz∂x_bkd)
        end
    end

    #  σxz
    for j in 2:nz-2
        for i in 3:nx-1

            # σxz
            ∂vx∂z_fwd = factz * (vx[i, j-1] - 27.0 * vx[i, j] + 27.0 * vx[i, j+1] - vx[i, j+2])
            ∂vz∂x_bkd = factx * (vz[i-2, j] - 27.0 * vz[i-1, j] + 27.0 * vz[i, j] - vz[i+1, j])

            ##=======================
            # C-PML stuff
            ##=======================
            # x boundaries
            if i <= halo
                # left boundary
                ψ_∂vz∂x[i, j] = b_x[i] * ψ_∂vz∂x[i, j] + a_x[i] * ∂vz∂x_bkd
                ∂vz∂x_bkd = ∂vz∂x_bkd + ψ_∂vz∂x[i, j]
            elseif i >= nx - halo + 1
                # right boundary
                ii = i - (nx - 2 * halo)
                ψ_∂vz∂x[ii, j] = b_x[ii] * ψ_∂vz∂x[ii, j] + a_x[ii] * ∂vz∂x_bkd
                ∂vz∂x_bkd = ∂vz∂x_bkd + ψ_∂vz∂x[ii, j]
            end
            # y boundaries
            if j <= halo && freetop == false
                # top boundary
                ψ_∂vx∂z[i, j] = b_z_half[j] * ψ_∂vx∂z[i, j] + a_z_half[j] * ∂vx∂z_fwd
                ∂vx∂z_fwd = ∂vx∂z_fwd + ψ_∂vx∂z[i, j]
            elseif j >= nz - halo + 1
                # bottom boundary
                jj = j - (nz - 2 * halo)
                ψ_∂vx∂z[i, jj] = b_z_half[jj] * ψ_∂vx∂z[i, jj] + a_z_half[jj] * ∂vx∂z_fwd
                ∂vx∂z_fwd = ∂vx∂z_fwd + ψ_∂vx∂z[i, jj]
            end
            ##=======================

            # # C-PML stuff
            # # DO NOT delete this part!
            # ψ_∂vz∂x[i,j] = b_x[i] * ψ_∂vz∂x[i,j] + a_x[i]*∂vz∂x_bkd
            # ψ_∂vx∂z[i,j] = b_z_half[j] * ψ_∂vx∂z[i,j] + a_z_half[j]*∂vx∂z_fwd
            # ∂vz∂x_bkd = ∂vz∂x_bkd + ψ_∂vz∂x[i,j]
            # ∂vx∂z_fwd = ∂vx∂z_fwd + ψ_∂vx∂z[i,j]

            # σxz
            σxz[i, j] = σxz[i, j] + μ_jhalf[i, j] * dt * (∂vx∂z_fwd + ∂vz∂x_bkd)
        end
    end
    return
end

function forward_onestep_CPML!(
    model::ElasticIsoCPMLWaveSimulation{T, N},
    srccoeij_bk::Array{Int},
    srccoeval_bk::Array{T},
    reccoeij_bk::Array{Int},
    reccoeval_bk::Array{T},
    srctf_bk::Matrix{T},
    traces_bk::Array{T},
    it::Int,
    Mxx_bk::Vector{T},
    Mzz_bk::Vector{T},
    Mxz_bk::Vector{T};
    save_trace::Bool=true
) where {T, N}
    # Extract info from grid
    freetop = model.cpmlparams.freeboundtop
    cpmlcoeffs = model.cpmlcoeffs
    dx = model.grid.spacing[1]
    dz = model.grid.spacing[2]
    dt = model.dt
    nx, nz = model.grid.size[1:2]
    halo = model.cpmlparams.halo
    grid = model.grid

    vx, vz = grid.fields["v"].value
    σxx, σzz, σxz = grid.fields["σ"].value

    ψ_∂σxx∂x, ψ_∂σxz∂x = grid.fields["ψ_∂σ∂x"].value
    ψ_∂σzz∂z, ψ_∂σxz∂z = grid.fields["ψ_∂σ∂z"].value
    ψ_∂vx∂x, ψ_∂vz∂x = grid.fields["ψ_∂v∂x"].value
    ψ_∂vx∂z, ψ_∂vz∂z = grid.fields["ψ_∂v∂z"].value

    a_x = cpmlcoeffs[1].a
    a_x_half = cpmlcoeffs[1].a_h
    b_x = cpmlcoeffs[1].b
    b_x_half = cpmlcoeffs[1].b_h

    a_z = cpmlcoeffs[2].a
    a_z_half = cpmlcoeffs[2].a_h
    b_z = cpmlcoeffs[2].b
    b_z_half = cpmlcoeffs[2].b_h

    ρ = grid.fields["ρ"].value
    ρ_ihalf_jhalf = grid.fields["ρ_ihalf_jhalf"].value
    λ_ihalf = grid.fields["λ_ihalf"].value
    μ_ihalf = grid.fields["μ_ihalf"].value
    μ_jhalf = grid.fields["μ_jhalf"].value

    # Precomputing divisions
    factx = 1.0 / (24.0 * dx)
    factz = 1.0 / (24.0 * dz)

    # update velocity vx 
    update_4thord_vx!(nx, nz, halo, vx, factx, factz, σxx, σxz, dt, ρ, ψ_∂σxx∂x, ψ_∂σxz∂z,
        b_x, b_z, a_x, a_z, freetop)
    # update velocity vz
    update_4thord_vz!(nx, nz, halo, vz, factx, factz, σxz, σzz, dt, ρ_ihalf_jhalf, ψ_∂σxz∂x,
        ψ_∂σzz∂z, b_x_half, b_z_half, a_x_half, a_z_half, freetop)

    # inject sources (external body force)
    #inject_bodyforce_sources!(vx,vz,fx,fz,srctf_bk, dt, possrcs_bk,it)

    # update stresses σxx and σzz 
    update_4thord_σxxσzz!(nx, nz, halo, σxx, σzz, factx, factz,
        vx, vz, dt, λ_ihalf, μ_ihalf,
        ψ_∂vx∂x, ψ_∂vz∂z,
        b_x_half, b_z, a_x_half, a_z, freetop)
    # update stress σxz
    update_4thord_σxz!(nx, nz, halo, σxz, factx, factz, vx, vz, dt,
        μ_jhalf, b_x, b_z_half,
        ψ_∂vx∂z, ψ_∂vz∂x, a_x, a_z_half, freetop)

    # inject sources (moment tensor type of internal force)
    inject_momten_sources2D!(σxx, σzz, σxz, Mxx_bk, Mzz_bk, Mxz_bk, srctf_bk, dt,
        srccoeij_bk, srccoeval_bk, it)
    #possrcs_bk,it)

    # record receivers
    if save_trace
        record_receivers2D!(vx, vz, traces_bk, reccoeij_bk, reccoeval_bk, it)
    end

    return
end

function adjoint_onestep_CPML!(
    model,
    srccoeij_bk,
    srccoeval_bk,
    residuals_bk,
    it
)
    # Extract info from grid
    freetop = model.cpmlparams.freeboundtop
    cpmlcoeffs = model.cpmlcoeffs
    dx = model.grid.spacing[1]
    dz = model.grid.spacing[2]
    dt = model.dt
    nx, nz = model.grid.size[1:2]
    halo = model.cpmlparams.halo
    grid = model.grid

    vx, vz = grid.fields["adjv"].value
    σxx, σzz, σxz = grid.fields["adjσ"].value

    ψ_∂σxx∂x, ψ_∂σxz∂x = grid.fields["adjψ_∂σ∂x"].value
    ψ_∂σzz∂z, ψ_∂σxz∂z = grid.fields["adjψ_∂σ∂z"].value
    ψ_∂vx∂x, ψ_∂vz∂x = grid.fields["adjψ_∂v∂x"].value
    ψ_∂vx∂z, ψ_∂vz∂z = grid.fields["adjψ_∂v∂z"].value

    a_x = cpmlcoeffs[1].a
    a_x_half = cpmlcoeffs[1].a_h
    b_x = cpmlcoeffs[1].b
    b_x_half = cpmlcoeffs[1].b_h

    a_z = cpmlcoeffs[2].a
    a_z_half = cpmlcoeffs[2].a_h
    b_z = cpmlcoeffs[2].b
    b_z_half = cpmlcoeffs[2].b_h

    ρ = grid.fields["ρ"].value
    ρ_ihalf_jhalf = grid.fields["ρ_ihalf_jhalf"].value
    λ_ihalf = grid.fields["λ_ihalf"].value
    μ_ihalf = grid.fields["μ_ihalf"].value
    μ_jhalf = grid.fields["μ_jhalf"].value

    # Precomputing divisions
    factx = 1.0 / (24.0 * dx)
    factz = 1.0 / (24.0 * dz)

    # update stresses σxx and σzz 
    update_4thord_σxxσzz!(nx, nz, halo, σxx, σzz, factx, factz,
        vx, vz, dt, λ_ihalf, μ_ihalf,
        ψ_∂vx∂x, ψ_∂vz∂z,
        b_x_half, b_z, a_x_half, a_z, freetop)
    # update stress σxz
    update_4thord_σxz!(nx, nz, halo, σxz, factx, factz, vx, vz, dt,
        μ_jhalf, b_x, b_z_half,
        ψ_∂vx∂z, ψ_∂vz∂x, a_x, a_z_half, freetop)
    
    # update velocity vx 
    update_4thord_vx!(nx, nz, halo, vx, factx, factz, σxx, σxz, dt, ρ, ψ_∂σxx∂x, ψ_∂σxz∂z,
        b_x, b_z, a_x, a_z, freetop)
    # update velocity vz
    update_4thord_vz!(nx, nz, halo, vz, factx, factz, σxz, σzz, dt, ρ_ihalf_jhalf, ψ_∂σxz∂x,
        ψ_∂σzz∂z, b_x_half, b_z_half, a_x_half, a_z_half, freetop)

    # inject sources (external body force)
    inject_vel_sources2D!(vx, vz, residuals_bk, srccoeij_bk, srccoeval_bk, ρ, ρ_ihalf_jhalf, it)


    return
end

function inject_momten_sources2D!(σxx, σzz, σxz, Mxx, Mzz, Mxz, srctf_bk, dt, srccoeij_bk, srccoeval_bk, it)
    #function inject_momten_sources!(σxx,σzz,σxz,Mxx, Mzz, Mxz, srctf_bk, dt, possrcs_bk, it)

    ## Inject the source as stress from moment tensor
    ##  See Igel 2017 Computational Seismology (book) page 31, 2.6.1
    lensrctf = length(srctf_bk)
    if it <= lensrctf
        # total number of interpolation points
        nsrcpts = size(srccoeij_bk, 1)
        # p runs on all interpolation points defined by the windowed sinc function
        # s runs on all actual source locations as specified by user input
        for p in 1:nsrcpts
            # [src_id, i, j]
            s, isrc, jsrc = srccoeij_bk[p, :]
            # update stresses on points computed from sinc interpolation 
            #     scaled with the coefficients' values
            σxx[isrc, jsrc] += Mxx[s] * srccoeval_bk[p] * srctf_bk[it]
            σzz[isrc, jsrc] += Mzz[s] * srccoeval_bk[p] * srctf_bk[it]
            σxz[isrc, jsrc] += Mxz[s] * srccoeval_bk[p] * srctf_bk[it]
        end

        # for s in axes(possrcs_bk, 1)
        #     isrc = possrcs_bk[s, 1]
        #     jsrc = possrcs_bk[s, 2]
        #     σxx[isrc,jsrc] += Mxx[s] * srctf_bk[it] * dt 
        #     σzz[isrc,jsrc] += Mzz[s] * srctf_bk[it] * dt 
        #     σxz[isrc,jsrc] += Mxz[s] * srctf_bk[it] * dt
        # end
    end
    return
end

function inject_vel_sources2D!(vx, vz, f, srccoeij_bk, srccoeval_bk, ρ, ρ_ihalf_jhalf, it)
    nsrcpts = size(srccoeij_bk, 1)
    for p in 1:nsrcpts
        s, isrc, jsrc = srccoeij_bk[p, 1], srccoeij_bk[p, 2], srccoeij_bk[p, 3]
        vx[isrc, jsrc] += srccoeval_bk[p] * f[it, 1, s] * dt / ρ[isrc, jsrc]
        vz[isrc, jsrc] += srccoeval_bk[p] * f[it, 2, s] * dt / ρ_ihalf_jhalf[isrc, jsrc]
    end
    return nothing
end

function record_receivers2D!(vx, vz, traces_bk, reccoeij_bk, reccoeval_bk, it)

    # total number of interpolation points
    nrecpts = size(reccoeij_bk, 1)
    # p runs on all interpolation points defined by the windowed sinc function
    # s runs on all actual source locations as specified by user input
    for p in 1:nrecpts
        # [src_id, i, j]
        r, irec, jrec = reccoeij_bk[p, :]
        # update traces by summing up values from all sinc interpolation points
        traces_bk[it, 1, r] += reccoeval_bk[r] * vx[irec, jrec]
        traces_bk[it, 2, r] += reccoeval_bk[r] * vz[irec, jrec]
    end

    # for ir in axes(posrecs, 1)
    #     irec = posrecs[ir, 1]
    #     jrec = posrecs[ir, 2]
    #     # interpolate velocities on the same grid?
    #     traces_bk[it,1,ir] = vx[irec, jrec]
    #     traces_bk[it,2,ir] = vz[irec, jrec]
    # end
    return
end

function correlate_gradient_ρ_kernel!(grad_ρ, adjv, v_curr, v_old, _dt)
    @. grad_ρ = grad_ρ + adjv * (v_old - v_curr) * _dt

    return nothing
end

function correlate_gradient_ihalf_kernel!(grad_λ_ihalf, grad_μ_ihalf, adjσxx, adjσzz, vx, vz, λ_ihalf, μ_ihalf, factx, factz, freetop, nx, nz)
    for j in 1:nz-1
        for i in 2:nx-2
            ∂vx∂x_fwd = ∂vz∂z_bkd = 0
            if freetop == true
                # j=1: we are on the free surface!
                if j == 1
                    # vx derivative only in x so no problem
                    ∂vx∂x_fwd = (vx[i-1, j] - 27.0 * vx[i, j] + 27.0 * vx[i+1, j] - vx[i+2, j]) * factx
                    # using boundary condition to calculate ∂vz∂z_bkd from ∂vx∂x_fwd
                    ∂vz∂z_bkd = -(λ_ihalf[i, j] / (λ_ihalf[i, j] + 2.0 * μ_ihalf[i, j])) * ∂vx∂x_fwd
                end
                # j=2: we are just below the surface (1/2)
                if j == 2
                    # vx derivative only in x so no problem
                    ∂vx∂x_fwd = (vx[i-1, j] - 27.0 * vx[i, j] + 27.0 * vx[i+1, j] - vx[i+2, j]) * factx
                    # zero velocity above the free surface
                    ∂vz∂z_bkd = (0.0 - 27.0 * vz[i, j-1] + 27.0 * vz[i, j] - vz[i, j+1]) * factz
                end
            end
            if j >= 3
                ∂vx∂x_fwd = (vx[i-1, j] - 27.0 * vx[i, j] + 27.0 * vx[i+1, j] - vx[i+2, j]) * factx
                ∂vz∂z_bkd = (vz[i, j-2] - 27.0 * vz[i, j-1] + 27.0 * vz[i, j] - vz[i, j+1]) * factz
            end
            # correlate
            grad_λ_ihalf[i, j] += -((∂vx∂x_fwd + ∂vz∂z_bkd) * (adjσxx[i, j] + adjσzz[i, j]))
            grad_μ_ihalf[i, j] += (-2 * ∂vx∂x_fwd * adjσxx[i, j]) + (-2 * ∂vz∂z_bkd * adjσzz[i, j])
        end
    end

    return nothing
end

function correlate_gradient_jhalf_kernel!(grad_μ_jhalf, adjσxz, vx, vz, factx, factz, freetop, nx, nz)
    for j in 1:nz-2
        for i in 3:nx-1
            ∂vx∂z_fwd = ∂vz∂x_bkd = 0
            if freetop
                if j == 1
                    # zero velocity above the free surface
                    ∂vx∂z_fwd = factz * (0.0 - 27.0 * vx[i, j] + 27.0 * vx[i, j+1] - vx[i, j+2])
                    # vz derivative only in x so no problem
                    ∂vz∂x_bkd = factx * (vz[i-2, j] - 27.0 * vz[i-1, j] + 27.0 * vz[i, j] - vz[i+1, j])
                end
            end
            if j >= 2
                ∂vx∂z_fwd = factz * (vx[i, j-1] - 27.0 * vx[i, j] + 27.0 * vx[i, j+1] - vx[i, j+2])
                ∂vz∂x_bkd = factx * (vz[i-2, j] - 27.0 * vz[i-1, j] + 27.0 * vz[i, j] - vz[i+1, j])
            end
            # correlate
            grad_μ_jhalf[i, j] += (-∂vx∂z_fwd-∂vz∂x_bkd) * adjσxz[i, j]
        end
    end

    return nothing
end

function correlate_gradients!(grid, vcurr, vold, dt, freetop)
    nx, nz = grid.size
    correlate_gradient_ρ_kernel!(grid.fields["grad_ρ"].value, grid.fields["adjv"].value[1], vcurr[1], vold[1], 1 / dt)
    correlate_gradient_ρ_kernel!(grid.fields["grad_ρ_ihalf_jhalf"].value, grid.fields["adjv"].value[2], vcurr[2], vold[2], 1 / dt)
    correlate_gradient_ihalf_kernel!(
        grid.fields["grad_λ_ihalf"].value,
        grid.fields["grad_μ_ihalf"].value,
        grid.fields["adjσ"].value[1], grid.fields["adjσ"].value[2], vcurr...,
        grid.fields["λ_ihalf"].value, grid.fields["μ_ihalf"].value,
        (1.0 ./ (24.0 .* grid.spacing))...,
        freetop, nx, nz
    )
    correlate_gradient_jhalf_kernel!(
        grid.fields["grad_μ_jhalf"].value,
        grid.fields["adjσ"].value[3], vcurr...,
        (1.0 ./ (24.0 .* grid.spacing))...,
        freetop, nx, nz
    )
end


#########################################
end  # end module
#########################################
