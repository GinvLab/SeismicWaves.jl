@parallel_indices (i, j) function update_4thord_vx!(
    nx, nz, halo, vx, factx, factz, σxx, σxz, dt, ρ_ihalf,
    ψ_∂σxx∂x, ψ_∂σxz∂z, b_x_half, b_z, a_x_half, a_z, freetop)
    if freetop && j == 1
        # on the free surface
        ∂σxx∂x = (-σxx[i+2, j] + 27 * σxx[i+1, j] - 27 * σxx[i, j] + σxx[i-1, j]) * factx
        # mirroring stresses at the top boundary
        σxz_top = -σxz[i, j]
        σxz_toptop = -σxz[i, j+1]
        ∂σxz∂z = (-σxz[i, j+1] + 27 * σxz[i, j] - 27 * σxz_top + σxz_toptop) * factz
    elseif freetop && j == 2
        # just below the free surface
        ∂σxx∂x = (-σxx[i+2, j] + 27 * σxx[i+1, j] - 27 * σxx[i, j] + σxx[i-1, j]) * factx
        # mirroring stress at the top boundary
        σxz_top = -σxz[i, j+1]
        ∂σxz∂z = (-σxz[i, j+1] + 27 * σxz[i, j] - 27 * σxz[i, j-1] + σxz_top) * factz
    elseif j >= 3
        ∂σxx∂x = (-σxx[i+2, j] + 27 * σxx[i+1, j] - 27 * σxx[i, j  ] + σxx[i-1, j]) * factx
        ∂σxz∂z = (-σxz[i, j+1] + 27 * σxz[i, j  ] - 27 * σxz[i, j-1] + σxz[i, j-2]) * factz
    else
        ∂σxx∂x = 0.0
        ∂σxz∂z = 0.0
    end

    ### CPML ###
    if i <= halo + 1
        # left boundary
        ψ_∂σxx∂x[i, j] = b_x_half[i] * ψ_∂σxx∂x[i, j] + a_x_half[i] * ∂σxx∂x
        ∂σxx∂x += ψ_∂σxx∂x[i, j]
    elseif i >= nx - halo - 1
        # right boundary
        ii = i - (nx - halo - 1) + 1 + (halo + 1)
        ψ_∂σxx∂x[ii, j] = b_x_half[ii] * ψ_∂σxx∂x[ii, j] + a_x_half[ii] * ∂σxx∂x
        ∂σxx∂x += ψ_∂σxx∂x[ii, j]
    end
    if j >= 2 && j <= halo + 1 && freetop == false
        # top boundary
        ψ_∂σxz∂z[i, j-1] = b_z[j-1] * ψ_∂σxz∂z[i, j-1] + a_z[j-1] * ∂σxz∂z
        ∂σxz∂z += ψ_∂σxz∂z[i, j-1]
    elseif j >= nz - halo
        # bottom boundary
        jj = j - (nz - halo) + 2 + (halo)
        ψ_∂σxz∂z[i, jj-1] = b_z[jj-1] * ψ_∂σxz∂z[i, jj-1] + a_z[jj-1] * ∂σxz∂z
        ∂σxz∂z += ψ_∂σxz∂z[i, jj-1]
    end
    ############

    vx[i, j] += dt * (∂σxx∂x + ∂σxz∂z) / ρ_ihalf[i, j]

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
    ρ_jhalf,
    ψ_∂σxz∂x,
    ψ_∂σzz∂z,
    b_x,
    b_z_half,
    a_x,
    a_z_half,
    freetop)
    if freetop && j == 1
        # just half step below the free surface
        # mirroring stress at the top boundary
        σzz_top = -σzz[i, j+2]
        ∂σzz∂z = (-σzz[i, j+2] + 27 * σzz[i, j+1] - 27 * σzz[i, j] + σzz_top) * factz
        ∂σxz∂x = (-σxz[i+1, j] + 27 * σxz[i, j] - 27 * σxz[i-1, j] + σxz[i-2, j]) * factx
    elseif j >= 2
        ∂σzz∂z = (-σzz[i, j+2] + 27 * σzz[i, j+1] - 27 * σzz[i, j] + σzz[i, j-1]) * factz
        ∂σxz∂x = (-σxz[i+1, j] + 27 * σxz[i, j] - 27 * σxz[i-1, j] + σxz[i-2, j]) * factx
    else
        ∂σzz∂z = 0.0
        ∂σxz∂x = 0.0
    end

    ### CPML ###
    if i <= halo + 1
        # left boundary
        ψ_∂σxz∂x[i-1, j] = b_x[i-1] * ψ_∂σxz∂x[i-1, j] + a_x[i-1] * ∂σxz∂x
        ∂σxz∂x += ψ_∂σxz∂x[i-1, j]
    elseif i >= nx - halo
        # right boundary
        ii = i - (nx - halo) + 2 + (halo)
        ψ_∂σxz∂x[ii-1, j] = b_x[ii-1] * ψ_∂σxz∂x[ii-1, j] + a_x[ii-1] * ∂σxz∂x
        ∂σxz∂x += ψ_∂σxz∂x[ii-1, j]
    end
    if j <= halo + 1 && freetop == false
        # top boundary
        ψ_∂σzz∂z[i, j] = b_z_half[j] * ψ_∂σzz∂z[i, j] + a_z_half[j] * ∂σzz∂z
        ∂σzz∂z += ψ_∂σzz∂z[i, j]
    elseif j >= nz - halo
        # bottom boundary
        jj = j - (nz - halo) + 2 + (halo + 1)
        ψ_∂σzz∂z[i, jj] = b_z_half[jj] * ψ_∂σzz∂z[i, jj] + a_z_half[jj] * ∂σzz∂z
        ∂σzz∂z += ψ_∂σzz∂z[i, jj]
    end
    ############

    vz[i, j] += dt * (∂σzz∂z + ∂σxz∂x) / ρ_jhalf[i, j]

    return nothing
end

@parallel_indices (i, j) function update_4thord_σxxσzz!(
    nx, nz, halo, σxx, σzz, factx, factz,
    vx, vz, dt, λ, μ, ψ_∂vx∂x, ψ_∂vz∂z,
    b_x, b_z, a_x, a_z, freetop)
    if freetop && j == 1
        # on the free surface, σzz = 0
        ∂vx∂x = (-vx[i+1, j] + 27 * vx[i, j] - 27 * vx[i-1, j] + vx[i-2, j]) * factx
        # using boundary condition to calculate ∂vz∂z
        ∂vz∂z = -(λ[i, j] / (λ[i, j] + 2*μ[i, j])) * ∂vx∂x
        # update stresses
        σxx[i, j] += dt * ((λ[i, j] + 2*μ[i, j]) * ∂vx∂x + λ[i, j] * ∂vz∂z)
        σzz[i, j] = 0.0
    elseif freetop && j == 2
        # just below the free surface
        ∂vx∂x = (-vx[i+1, j] + 27 * vx[i, j] - 27 * vx[i-1, j] + vx[i-2, j]) * factx
        # zero velocity above the free surface
        ∂vz∂z = (-vz[i, j+1] + 27 * vz[i, j] - 27 * vz[i, j-1] + 0.0) * factz
        # update stresses
        σxx[i, j] += dt * ((λ[i, j] + 2*μ[i, j]) * ∂vx∂x + λ[i, j] * ∂vz∂z)
        σzz[i, j] += dt * ((λ[i, j] + 2*μ[i, j]) * ∂vz∂z + λ[i, j] * ∂vx∂x)
    elseif j >= 3
        ∂vx∂x = (-vx[i+1, j] + 27 * vx[i, j] - 27 * vx[i-1, j] + vx[i-2, j]) * factx
        ∂vz∂z = (-vz[i, j+1] + 27 * vz[i, j] - 27 * vz[i, j-1] + vz[i, j-2]) * factz

        ### CPML ###
        if i <= halo + 1
            # left boundary
            ψ_∂vx∂x[i-1, j] = b_x[i-1] * ψ_∂vx∂x[i-1, j] + a_x[i-1] * ∂vx∂x
            ∂vx∂x += ψ_∂vx∂x[i-1, j]
        elseif i >= nx - halo
            # right boundary
            ii = i - (nx - halo) + 2 + (halo)
            ψ_∂vx∂x[ii-1, j] = b_x[ii-1] * ψ_∂vx∂x[ii-1, j] + a_x[ii-1] * ∂vx∂x
            ∂vx∂x += ψ_∂vx∂x[ii-1, j]
        end
        if j <= halo + 1 && freetop == false
            # top boundary
            ψ_∂vz∂z[i, j-1] = b_z[j-1] * ψ_∂vz∂z[i, j-1] + a_z[j-1] * ∂vz∂z
            ∂vz∂z += ψ_∂vz∂z[i, j-1]
        elseif j >= nz - halo
            # bottom boundary
            jj = j - (nz - halo) + 2 + (halo)
            ψ_∂vz∂z[i, jj-1] = b_z[jj-1] * ψ_∂vz∂z[i, jj-1] + a_z[jj-1] * ∂vz∂z
            ∂vz∂z += ψ_∂vz∂z[i, jj-1]
        end
        ############

        # update stresses
        σxx[i, j] += dt * ((λ[i, j] + 2*μ[i, j]) * ∂vx∂x +  λ[i, j]              * ∂vz∂z)
        σzz[i, j] += dt * ( λ[i, j]              * ∂vx∂x + (λ[i, j] + 2*μ[i, j]) * ∂vz∂z)
    end

    return nothing
end

@parallel_indices (i, j) function update_4thord_σxz!(
    nx, nz, halo, σxz, factx, factz, vx, vz, dt, μ_ihalf_jhalf,
    ψ_∂vx∂z, ψ_∂vz∂x, b_x_half, b_z_half, a_x_half, a_z_half, freetop)
    if freetop && j == 1
        # zero velocity above the free surface
        ∂vx∂z = (-vx[i, j+2] + 27 * vx[i, j+1] - 27 * vx[i, j] + 0.0       ) * factz
        ∂vz∂x = (-vz[i+2, j] + 27 * vz[i+1, j] - 27 * vz[i, j] + vz[i-1, j]) * factx
    elseif j >= 2
        ∂vx∂z = (-vx[i, j+2] + 27 * vx[i, j+1] - 27 * vx[i, j] + vx[i, j-1]) * factz
        ∂vz∂x = (-vz[i+2, j] + 27 * vz[i+1, j] - 27 * vz[i, j] + vz[i-1, j]) * factx
    else
        ∂vx∂z = 0.0
        ∂vz∂x = 0.0
    end

    ### CPML ###
    if i <= halo + 1
        # left boundary
        ψ_∂vz∂x[i, j] = b_x_half[i] * ψ_∂vz∂x[i, j] + a_x_half[i] * ∂vz∂x
        ∂vz∂x += ψ_∂vz∂x[i, j]
    elseif i >= nx - halo - 1
        # right boundary
        ii = i - (nx - halo - 1) + 1 + (halo + 1)
        ψ_∂vz∂x[ii, j] = b_x_half[ii] * ψ_∂vz∂x[ii, j] + a_x_half[ii] * ∂vz∂x
        ∂vz∂x += ψ_∂vz∂x[ii, j]
    end
    if j <= halo + 1 && freetop == false
        # top boundary
        ψ_∂vx∂z[i, j] = b_z_half[j] * ψ_∂vx∂z[i, j] + a_z_half[j] * ∂vx∂z
        ∂vx∂z += ψ_∂vx∂z[i, j]
    elseif j >= nz - halo - 1
        # bottom boundary
        jj = j - (nz - halo - 1) + 1 + (halo + 1)
        ψ_∂vx∂z[i, jj] = b_z_half[jj] * ψ_∂vx∂z[i, jj] + a_z_half[jj] * ∂vx∂z
        ∂vx∂z += ψ_∂vx∂z[i, jj]
    end
    ############

    σxz[i, j] += dt * μ_ihalf_jhalf[i, j] * (∂vx∂z + ∂vz∂x)

    return nothing
end

@parallel_indices (p) function inject_momten_sources2D_σxx_σzz!(σxx, σzz, Mxx, Mzz, srctf_bk, srccoeij_bk, srccoeval_bk, it, s, dt)
    isrc, jsrc = srccoeij_bk[p, 1], srccoeij_bk[p, 2]
    σxx[isrc, jsrc] += Mxx[s] * srccoeval_bk[p] * srctf_bk[it, s] * dt
    σzz[isrc, jsrc] += Mzz[s] * srccoeval_bk[p] * srctf_bk[it, s] * dt

    return nothing
end

@parallel_indices (p) function inject_momten_sources2D_σxz!(σxz, Mxz, srctf_bk, srccoeij_bk, srccoeval_bk, it, s, dt)
    isrc, jsrc = srccoeij_bk[p, 1], srccoeij_bk[p, 2]
    σxz[isrc, jsrc] += Mxz[s] * srccoeval_bk[p] * srctf_bk[it, s] * dt

    return nothing
end

@parallel_indices (p) function inject_external_sources2D_vx!(vx, srctf_bk, srccoeij_bk, srccoeval_bk, ρ_ihalf, it, s, dt)
    isrc, jsrc = srccoeij_bk[p, 1], srccoeij_bk[p, 2]
    vx[isrc, jsrc] += srccoeval_bk[p] * srctf_bk[it, 1, s] / ρ_ihalf[isrc, jsrc] * dt
    return nothing
end

@parallel_indices (p) function inject_external_sources2D_vz!(vz, srctf_bk, srccoeij_bk, srccoeval_bk, ρ_jhalf, it, s, dt)
    isrc, jsrc = srccoeij_bk[p, 1], srccoeij_bk[p, 2]
    vz[isrc, jsrc] += srccoeval_bk[p] * srctf_bk[it, 2, s] / ρ_jhalf[isrc, jsrc] * dt
    return nothing
end

@parallel_indices (p) function record_receivers2D_vx!(vx, traces_vx_bk_buf, reccoeij_vx, reccoeval_vx)
    irec, jrec = reccoeij_vx[p, 1], reccoeij_vx[p, 2]
    traces_vx_bk_buf[p] = reccoeval_vx[p] * vx[irec, jrec]
    return nothing
end

@parallel_indices (p) function record_receivers2D_vz!(vz, traces_vz_bk_buf, reccoeij_vz, reccoeval_vz)
    irec, jrec = reccoeij_vz[p, 1], reccoeij_vz[p, 2]
    traces_vz_bk_buf[p] = reccoeval_vz[p] * vz[irec, jrec]
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
    reduced_buf,
    traces_vx_bk_buf,
    traces_vz_bk_buf,
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

    λ = grid.fields["λ"].value
    μ = grid.fields["μ"].value
    ρ_ihalf = grid.fields["ρ_ihalf"].value
    ρ_jhalf = grid.fields["ρ_jhalf"].value
    μ_ihalf_jhalf = grid.fields["μ_ihalf_jhalf"].value

    # Precomputing divisions
    factx = 1.0 / (24.0 * dx)
    factz = 1.0 / (24.0 * dz)

    # update velocity vx
    @parallel (2:nx-2, 1:nz-2) update_4thord_vx!(nx, nz, halo, vx, factx, factz, σxx, σxz, dt, ρ_ihalf,
                                                 ψ_∂σxx∂x, ψ_∂σxz∂z, b_x_half, b_z, a_x_half, a_z, freetop)
    # update velocity vz
    @parallel (3:nx-2, 1:nz-2) update_4thord_vz!(nx, nz, halo, vz, factx, factz, σxz, σzz, dt, ρ_jhalf,
                                                 ψ_∂σxz∂x, ψ_∂σzz∂z, b_x, b_z_half, a_x, a_z_half, freetop)

    # record receivers
    if save_trace
        nrecs = size(reccoeij_vx, 1)
        for r in 1:nrecs
            nrecspts_vx = size(reccoeij_vx[r], 1)
            @parallel (1:nrecspts_vx) record_receivers2D_vx!(vx, traces_vx_bk_buf[r], reccoeij_vx[r], reccoeval_vx[r])
            reduced_buf[r] = reduce(+, traces_vx_bk_buf[r])
        end
        copyto!(@view(traces_bk[it, 1, :]), reduced_buf)
        for r in 1:nrecs
            nrecspts_vz = size(reccoeij_vz[r], 1)
            @parallel (1:nrecspts_vz) record_receivers2D_vz!(vz, traces_vz_bk_buf[r], reccoeij_vz[r], reccoeval_vz[r])
            reduced_buf[r] = reduce(+, traces_vz_bk_buf[r])
        end
        copyto!(@view(traces_bk[it, 2, :]), reduced_buf)
    end

    # update stresses σxx and σzz 
    @parallel (3:nx-2, 1:nz-2) update_4thord_σxxσzz!(nx, nz, halo, σxx, σzz, factx, factz, vx, vz, dt, λ, μ,
                                                     ψ_∂vx∂x, ψ_∂vz∂z, b_x, b_z, a_x, a_z, freetop)
    # update stress σxz
    @parallel (2:nx-2, 1:nz-2) update_4thord_σxz!(nx, nz, halo, σxz, factx, factz, vx, vz, dt, μ_ihalf_jhalf,
                                                  ψ_∂vx∂z, ψ_∂vz∂x, b_x_half, b_z_half, a_x_half, a_z_half, freetop)

    # inject sources (moment tensor type of internal force)
    nrecs_xx = size(srccoeij_xx, 1)
    nrecs_xz = size(srccoeij_xz, 1)
    for s in 1:nrecs_xx
        nsrcpts_xx = size(srccoeij_xx[s], 1)
        @parallel (1:nsrcpts_xx) inject_momten_sources2D_σxx_σzz!(σxx, σzz, Mxx_bk, Mzz_bk, srctf_bk, srccoeij_xx[s], srccoeval_xx[s], it, s, dt)
    end
    for s in 1:nrecs_xz
        nsrcpts_xz = size(srccoeij_xz[s], 1)
        @parallel (1:nsrcpts_xz) inject_momten_sources2D_σxz!(σxz, Mxz_bk, srctf_bk, srccoeij_xz[s], srccoeval_xz[s], it, s, dt)
    end

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
    reduced_buf,
    traces_vx_bk_buf,
    traces_vz_bk_buf,
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

    λ = grid.fields["λ"].value
    μ = grid.fields["μ"].value
    ρ_ihalf = grid.fields["ρ_ihalf"].value
    ρ_jhalf = grid.fields["ρ_jhalf"].value
    μ_ihalf_jhalf = grid.fields["μ_ihalf_jhalf"].value

    # Precomputing divisions
    factx = 1.0 / (24.0 * dx)
    factz = 1.0 / (24.0 * dz)

    # update velocity vx
    @parallel (2:nx-2, 1:nz-2) update_4thord_vx!(nx, nz, halo, vx, factx, factz, σxx, σxz, dt, ρ_ihalf,
                                                 ψ_∂σxx∂x, ψ_∂σxz∂z, b_x_half, b_z, a_x_half, a_z, freetop)
    # update velocity vz
    @parallel (3:nx-2, 1:nz-2) update_4thord_vz!(nx, nz, halo, vz, factx, factz, σxz, σzz, dt, ρ_jhalf,
                                                 ψ_∂σxz∂x, ψ_∂σzz∂z, b_x, b_z_half, a_x, a_z_half, freetop)

    # inject sources
    nsrcs_vx = size(srccoeij_vx, 1)
    nsrcs_vz = size(srccoeij_vz, 1)
    for s in 1:nsrcs_vx
        nsrcpts_vx = size(srccoeij_vx[s], 1)
        @parallel (1:nsrcpts_vx) inject_external_sources2D_vx!(vx, srctf_bk, srccoeij_vx[s], srccoeval_vx[s], ρ_ihalf, it, s, dt)
    end
    for s in 1:nsrcs_vz
        nsrcpts_vz = size(srccoeij_vz[s], 1)
        @parallel (1:nsrcpts_vz) inject_external_sources2D_vz!(vz, srctf_bk, srccoeij_vz[s], srccoeval_vz[s], ρ_jhalf, it, s, dt)
    end

    # record receivers
    if save_trace
        nrecs = size(reccoeij_vx, 1)
        for r in 1:nrecs
            nrecspts_vx = size(reccoeij_vx[r], 1)
            @parallel (1:nrecspts_vx) record_receivers2D_vx!(vx, traces_vx_bk_buf[r], reccoeij_vx[r], reccoeval_vx[r])
            reduced_buf[r] = reduce(+, traces_vx_bk_buf[r])
        end
        copyto!(@view(traces_bk[it, 1, :]), reduced_buf)
        for r in 1:nrecs
            nrecspts_vz = size(reccoeij_vz[r], 1)
            @parallel (1:nrecspts_vz) record_receivers2D_vz!(vz, traces_vz_bk_buf[r], reccoeij_vz[r], reccoeval_vz[r])
            reduced_buf[r] = reduce(+, traces_vz_bk_buf[r])
        end
        copyto!(@view(traces_bk[it, 2, :]), reduced_buf)
    end

    # update stresses σxx and σzz 
    @parallel (3:nx-2, 1:nz-2) update_4thord_σxxσzz!(nx, nz, halo, σxx, σzz, factx, factz, vx, vz, dt, λ, μ,
                                                     ψ_∂vx∂x, ψ_∂vz∂z, b_x, b_z, a_x, a_z, freetop)
    # update stress σxz
    @parallel (2:nx-2, 1:nz-2) update_4thord_σxz!(nx, nz, halo, σxz, factx, factz, vx, vz, dt, μ_ihalf_jhalf,
                                                  ψ_∂vx∂z, ψ_∂vz∂x, b_x_half, b_z_half, a_x_half, a_z_half, freetop)

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

    λ = grid.fields["λ"].value
    μ = grid.fields["μ"].value
    ρ_ihalf = grid.fields["ρ_ihalf"].value
    ρ_jhalf = grid.fields["ρ_jhalf"].value
    μ_ihalf_jhalf = grid.fields["μ_ihalf_jhalf"].value

    # Precomputing divisions
    factx = 1.0 / (24.0 * dx)
    factz = 1.0 / (24.0 * dz)
    
    # update stresses σxx and σzz 
    @parallel (3:nx-2, 1:nz-2) update_4thord_σxxσzz!(nx, nz, halo, σxx, σzz, factx, factz, vx, vz, dt, λ, μ,
                                                     ψ_∂vx∂x, ψ_∂vz∂z, b_x, b_z, a_x, a_z, freetop)
    # update stress σxz
    @parallel (2:nx-2, 1:nz-2) update_4thord_σxz!(nx, nz, halo, σxz, factx, factz, vx, vz, dt, μ_ihalf_jhalf,
                                                  ψ_∂vx∂z, ψ_∂vz∂x, b_x_half, b_z_half, a_x_half, a_z_half, freetop)

    # update velocity vx
    @parallel (2:nx-2, 1:nz-2) update_4thord_vx!(nx, nz, halo, vx, factx, factz, σxx, σxz, dt, ρ_ihalf,
                                                 ψ_∂σxx∂x, ψ_∂σxz∂z, b_x_half, b_z, a_x_half, a_z, freetop)
    # update velocity vz
    @parallel (3:nx-2, 1:nz-2) update_4thord_vz!(nx, nz, halo, vz, factx, factz, σxz, σzz, dt, ρ_jhalf,
                                                 ψ_∂σxz∂x, ψ_∂σzz∂z, b_x, b_z_half, a_x, a_z_half, freetop)

    # inject sources (residuals as velocities)
    nsrcs_vx = size(srccoeij_vx, 1)
    nsrcs_vz = size(srccoeij_vz, 1)
    for s in 1:nsrcs_vx
        nsrcpts_vx = size(srccoeij_vx[s], 1)
        @parallel (1:nsrcpts_vx) inject_external_sources2D_vx!(vx, residuals_bk, srccoeij_vx[s], srccoeval_vx[s], ρ_ihalf, it, s, dt)
    end
    for s in 1:nsrcs_vz
        nsrcpts_vz = size(srccoeij_vz[s], 1)
        @parallel (1:nsrcpts_vz) inject_external_sources2D_vz!(vz, residuals_bk, srccoeij_vz[s], srccoeval_vz[s], ρ_jhalf, it, s, dt)
    end

    return
end