@parallel_indices (i, j) function update_4thord_vx!(nx, nz, halo, vx, factx, factz, σxx, σxz, dt, ρ, ψ_∂σxx∂x, ψ_∂σxz∂z, b_x, b_z, a_x, a_z, freetop)
    if freetop && j <= 2
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

    if j >= 3
        ∂σxx∂x_bkw = factx * (σxx[i-2, j] - 27.0 * σxx[i-1, j] + 27.0 * σxx[i, j] - σxx[i+1, j])
        ∂σxz∂z_bkw = factz * (σxz[i, j-2] - 27.0 * σxz[i, j-1] + 27.0 * σxz[i, j] - σxz[i, j+1])

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

        # update velocity
        vx[i, j] = vx[i, j] + (dt / ρ[i, j]) * (∂σxx∂x_bkw + ∂σxz∂z_bkw)
    end

    return nothing
end

@parallel_indices (i, j) function update_4thord_vz!(
    nx,
    nz,
    halo,
    vz,
    factx,
    factz,
    σxz,
    σzz,
    dt,
    ρ_ihalf_jhalf,
    ψ_∂σxz∂x,
    ψ_∂σzz∂z,
    b_x_half,
    b_z_half,
    a_x_half,
    a_z_half,
    freetop
)
    if freetop && j <= 2
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

    #  vz       
    if j >= 3
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

        # update velocity (ρ has been interpolated in advance)
        vz[i, j] = vz[i, j] + (dt / ρ_ihalf_jhalf[i, j]) * (∂σxz∂x_fwd + ∂σzz∂z_fwd)
    end

    return nothing
end

@parallel_indices (i, j) function update_4thord_σxxσzz!(nx, nz, halo, σxx, σzz, factx, factz,
    vx, vz, dt, λ_ihalf, μ_ihalf, ψ_∂vx∂x, ψ_∂vz∂z,
    b_x_half, b_z, a_x_half, a_z, freetop)
    if freetop == true
        # σxx, σzz
        # j=1: we are on the free surface!
        if j == 1
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
        if j == 2
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
    if j >= 3
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
        # σxx
        σxx[i, j] = σxx[i, j] + (λ_ihalf[i, j] + 2.0 * μ_ihalf[i, j]) * dt * ∂vx∂x_fwd +
                    λ_ihalf[i, j] * dt * ∂vz∂z_bkd

        ## derivatives are the same than for σxx 
        # σzz
        σzz[i, j] = σzz[i, j] + (λ_ihalf[i, j] + 2.0 * μ_ihalf[i, j]) * dt * ∂vz∂z_bkd +
                    λ_ihalf[i, j] * dt * ∂vx∂x_fwd
    end

    return nothing
end

@parallel_indices (i, j) function update_4thord_σxz!(nx, nz, halo, σxz, factx, factz, vx, vz, dt,
    μ_jhalf, b_x, b_z_half,
    ψ_∂vx∂z, ψ_∂vz∂x, a_x, a_z_half,
    freetop)
    if freetop
        # σxz
        if j == 1
            # zero velocity above the free surface
            ∂vx∂z_fwd = factz * (0.0 - 27.0 * vx[i, j] + 27.0 * vx[i, j+1] - vx[i, j+2])
            # vz derivative only in x so no problem
            ∂vz∂x_bkd = factx * (vz[i-2, j] - 27.0 * vz[i-1, j] + 27.0 * vz[i, j] - vz[i+1, j])
            # σxz
            σxz[i, j] = σxz[i, j] + μ_jhalf[i, j] * dt * (∂vx∂z_fwd + ∂vz∂x_bkd)
        end
    end

    #  σxz
    if j >= 2
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

        # σxz
        σxz[i, j] = σxz[i, j] + μ_jhalf[i, j] * dt * (∂vx∂z_fwd + ∂vz∂x_bkd)
    end

    return nothing
end

@parallel_indices (p) function inject_momten_sources2D_σxx_σzz!(σxx, σzz, Mxx, Mzz, srctf_bk, srccoeij_bk, srccoeval_bk, it)
    s, isrc, jsrc = srccoeij_bk[p, 1], srccoeij_bk[p, 2], srccoeij_bk[p, 3]
    σxx[isrc, jsrc] += Mxx[s] * srccoeval_bk[p] * srctf_bk[it, s]
    σzz[isrc, jsrc] += Mzz[s] * srccoeval_bk[p] * srctf_bk[it, s]

    return nothing
end

@parallel_indices (p) function inject_momten_sources2D_σxz!(σxz, Mxz, srctf_bk, srccoeij_bk, srccoeval_bk, it)
    s, isrc, jsrc = srccoeij_bk[p, 1], srccoeij_bk[p, 2], srccoeij_bk[p, 3]
    σxz[isrc, jsrc] += Mxz[s] * srccoeval_bk[p] * srctf_bk[it, s]

    return nothing
end

@parallel_indices (p) function inject_external_sources2D_vx!(vx, srctf_bk, srccoeij_bk, srccoeval_bk, ρ, it)
    s, isrc, jsrc = srccoeij_bk[p, 1], srccoeij_bk[p, 2], srccoeij_bk[p, 3]
    vx[isrc, jsrc] += srccoeval_bk[p] * srctf_bk[it, 1, s] / ρ[isrc, jsrc]
    return nothing
end

@parallel_indices (p) function inject_external_sources2D_vz!(vz, srctf_bk, srccoeij_bk, srccoeval_bk, ρ_ihalf_jhalf, it)
    s, isrc, jsrc = srccoeij_bk[p, 1], srccoeij_bk[p, 2], srccoeij_bk[p, 3]
    vz[isrc, jsrc] += srccoeval_bk[p] * srctf_bk[it, 2, s] / ρ_ihalf_jhalf[isrc, jsrc]
    return nothing
end

@parallel_indices (p) function record_receivers2D_vx!(vx, traces_bk, reccoeij_vx, reccoeval_vx, it)
    r, irec, jrec = reccoeij_vx[p, 1], reccoeij_vx[p, 2], reccoeij_vx[p, 3]
    traces_bk[it, 1, r] += reccoeval_vx[p] * vx[irec, jrec]
    return nothing
end

@parallel_indices (p) function record_receivers2D_vz!(vz, traces_bk, reccoeij_vz, reccoeval_vz, it)
    r, irec, jrec = reccoeij_vz[p, 1], reccoeij_vz[p, 2], reccoeij_vz[p, 3]
    traces_bk[it, 2, r] += reccoeval_vz[p] * vz[irec, jrec]
    return nothing
end

function forward_onestep_CPML!(
    model,
    srccoeij_xx,
    srccoeval_xx,
    srccoeij_xz,
    srccoeval_xz,
    reccoeij_vx,
    reccoeval_vx,
    reccoeij_vz,
    reccoeval_vz,
    srctf_bk,
    traces_bk,
    it::Int,
    Mxx_bk,
    Mzz_bk,
    Mxz_bk;
    save_trace::Bool=true
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
    @parallel (3:nx-1, 1:nz-1) update_4thord_vx!(nx, nz, halo, vx, factx, factz, σxx, σxz, dt, ρ, ψ_∂σxx∂x, ψ_∂σxz∂z,
        b_x, b_z, a_x, a_z, freetop)
    # update velocity vz
    @parallel (2:nx-2, 1:nz-2) update_4thord_vz!(nx, nz, halo, vz, factx, factz, σxz, σzz, dt, ρ_ihalf_jhalf, ψ_∂σxz∂x,
        ψ_∂σzz∂z, b_x_half, b_z_half, a_x_half, a_z_half, freetop)

    # record receivers
    if save_trace
        nrecpts_vx = size(reccoeij_vx, 1)
        nrecpts_vz = size(reccoeij_vz, 1)
        @parallel (1:nrecpts_vx) record_receivers2D_vx!(vx, traces_bk, reccoeij_vx, reccoeval_vx, it)
        @parallel (1:nrecpts_vz) record_receivers2D_vz!(vz, traces_bk, reccoeij_vz, reccoeval_vz, it)
    end

    # update stresses σxx and σzz 
    @parallel (2:nx-2, 1:nz-1) update_4thord_σxxσzz!(nx, nz, halo, σxx, σzz, factx, factz,
        vx, vz, dt, λ_ihalf, μ_ihalf,
        ψ_∂vx∂x, ψ_∂vz∂z,
        b_x_half, b_z, a_x_half, a_z, freetop)
    # update stress σxz
    @parallel (3:nx-1, 1:nz-2) update_4thord_σxz!(nx, nz, halo, σxz, factx, factz, vx, vz, dt,
        μ_jhalf, b_x, b_z_half,
        ψ_∂vx∂z, ψ_∂vz∂x, a_x, a_z_half, freetop)

    # inject sources (moment tensor type of internal force)
    nsrcpts_xx = size(srccoeij_xx, 1)
    nsrcpts_xz = size(srccoeij_xz, 1)
    @parallel (1:nsrcpts_xx) inject_momten_sources2D_σxx_σzz!(σxx, σzz, Mxx_bk, Mzz_bk, srctf_bk, srccoeij_xx, srccoeval_xx, it)
    @parallel (1:nsrcpts_xz) inject_momten_sources2D_σxz!(σxz, Mxz_bk, srctf_bk, srccoeij_xz, srccoeval_xz, it)

    return
end

function forward_onestep_CPML!(
    model,
    srccoeij_vx,
    srccoeval_vx,
    srccoeij_vz,
    srccoeval_vz,
    reccoeij_vx,
    reccoeval_vx,
    reccoeij_vz,
    reccoeval_vz,
    srctf_bk,
    traces_bk,
    it::Int;
    save_trace::Bool=true
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
    @parallel (3:nx-1, 1:nz-1) update_4thord_vx!(nx, nz, halo, vx, factx, factz, σxx, σxz, dt, ρ, ψ_∂σxx∂x, ψ_∂σxz∂z,
        b_x, b_z, a_x, a_z, freetop)
    # update velocity vz
    @parallel (2:nx-2, 1:nz-2) update_4thord_vz!(nx, nz, halo, vz, factx, factz, σxz, σzz, dt, ρ_ihalf_jhalf, ψ_∂σxz∂x,
        ψ_∂σzz∂z, b_x_half, b_z_half, a_x_half, a_z_half, freetop)

    # inject sources (residuals as velocities)
    nsrcpts_vx = size(srccoeij_vx, 1)
    nsrcpts_vz = size(srccoeij_vz, 1)
    @parallel (1:nsrcpts_vx) inject_external_sources2D_vx!(vx, srctf_bk, srccoeij_vx, srccoeval_vx, ρ, it)
    @parallel (1:nsrcpts_vz) inject_external_sources2D_vz!(vz, srctf_bk, srccoeij_vz, srccoeval_vz, ρ_ihalf_jhalf, it)

    # record receivers
    if save_trace
        nrecpts_vx = size(reccoeij_vx, 1)
        nrecpts_vz = size(reccoeij_vz, 1)
        @parallel (1:nrecpts_vx) record_receivers2D_vx!(vx, traces_bk, reccoeij_vx, reccoeval_vx, it)
        @parallel (1:nrecpts_vz) record_receivers2D_vz!(vz, traces_bk, reccoeij_vz, reccoeval_vz, it)
    end

    # update stresses σxx and σzz 
    @parallel (2:nx-2, 1:nz-1) update_4thord_σxxσzz!(nx, nz, halo, σxx, σzz, factx, factz,
        vx, vz, dt, λ_ihalf, μ_ihalf,
        ψ_∂vx∂x, ψ_∂vz∂z,
        b_x_half, b_z, a_x_half, a_z, freetop)
    # update stress σxz
    @parallel (3:nx-1, 1:nz-2) update_4thord_σxz!(nx, nz, halo, σxz, factx, factz, vx, vz, dt,
        μ_jhalf, b_x, b_z_half,
        ψ_∂vx∂z, ψ_∂vz∂x, a_x, a_z_half, freetop)

    return
end

function adjoint_onestep_CPML!(
    model,
    srccoeij_vx,
    srccoeval_vx,
    srccoeij_vz,
    srccoeval_vz,
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
    @parallel (2:nx-2, 1:nz-1) update_4thord_σxxσzz!(nx, nz, halo, σxx, σzz, factx, factz,
        vx, vz, dt, λ_ihalf, μ_ihalf,
        ψ_∂vx∂x, ψ_∂vz∂z,
        b_x_half, b_z, a_x_half, a_z, freetop)
    # update stress σxz
    @parallel (3:nx-1, 1:nz-2) update_4thord_σxz!(nx, nz, halo, σxz, factx, factz, vx, vz, dt,
        μ_jhalf, b_x, b_z_half,
        ψ_∂vx∂z, ψ_∂vz∂x, a_x, a_z_half, freetop)

    # update velocity vx 
    @parallel (3:nx-1, 1:nz-1) update_4thord_vx!(nx, nz, halo, vx, factx, factz, σxx, σxz, dt, ρ, ψ_∂σxx∂x, ψ_∂σxz∂z,
        b_x, b_z, a_x, a_z, freetop)
    # update velocity vz
    @parallel (2:nx-2, 1:nz-2) update_4thord_vz!(nx, nz, halo, vz, factx, factz, σxz, σzz, dt, ρ_ihalf_jhalf, ψ_∂σxz∂x,
        ψ_∂σzz∂z, b_x_half, b_z_half, a_x_half, a_z_half, freetop)

    # inject sources (residuals as velocities)
    nsrcpts_vx = size(srccoeij_vx, 1)
    nsrcpts_vz = size(srccoeij_vz, 1)
    @parallel (1:nsrcpts_vx) inject_external_sources2D_vx!(vx, residuals_bk, srccoeij_vx, srccoeval_vx, ρ, it)
    @parallel (1:nsrcpts_vz) inject_external_sources2D_vz!(vz, residuals_bk, srccoeij_vz, srccoeval_vz, ρ_ihalf_jhalf, it)

    return
end