@parallel_indices (i, j, k) function update_ψ_x!(
    ψ_x_l, ψ_x_r, pcur,
    halo, _dx, nx,
    a_x_hl, a_x_hr,
    b_K_x_hl, b_K_x_hr
)
    ii = i + nx - halo - 2  # shift for right boundary pressure indices
    # left boundary
    ψ_x_l[i, j, k] = b_K_x_hl[i] * ψ_x_l[i, j, k] + a_x_hl[i] * (pcur[i+1, j, k] - pcur[i, j, k]) * _dx
    # right boundary
    ψ_x_r[i, j, k] = b_K_x_hr[i] * ψ_x_r[i, j, k] + a_x_hr[i] * (pcur[ii+1, j, k] - pcur[ii, j, k]) * _dx

    return nothing
end

@parallel_indices (i, j, k) function update_ψ_y!(
    ψ_y_l, ψ_y_r, pcur,
    halo, _dy, ny,
    a_y_hl, a_y_hr,
    b_K_y_hl, b_K_y_hr
)
    jj = j + ny - halo - 2  # shift for bottom boundary pressure indices
    # top boundary
    ψ_y_l[i, j, k] = b_K_y_hl[j] * ψ_y_l[i, j, k] + a_y_hl[j] * (pcur[i, j+1, k] - pcur[i, j, k]) * _dy
    # bottom boundary
    ψ_y_r[i, j, k] = b_K_y_hr[j] * ψ_y_r[i, j, k] + a_y_hr[j] * (pcur[i, jj+1, k] - pcur[i, jj, k]) * _dy

    return nothing
end

@parallel_indices (i, j, k) function update_ψ_z!(
    ψ_z_l, ψ_z_r, pcur,
    halo, _dz, nz,
    a_z_hl, a_z_hr,
    b_K_z_hl, b_K_z_hr
)
    kk = k + nz - halo - 2  # shift for bottom boundary pressure indices
    # front boundary
    ψ_z_l[i, j, k] = b_K_z_hl[k] * ψ_z_l[i, j, k] + a_z_hl[k] * (pcur[i, j, k+1] - pcur[i, j, k]) * _dz
    # back boundary
    ψ_z_r[i, j, k] = b_K_z_hr[k] * ψ_z_r[i, j, k] + a_z_hr[k] * (pcur[i, j, kk+1] - pcur[i, j, kk]) * _dz

    return nothing
end

@parallel_indices (i, j, k) function update_p_CPML!(
    pold, pcur, pnew, halo, fact,
    _dx, _dx2, _dy, _dy2, _dz, _dz2, nx, ny,
    nz,
    ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r, ψ_z_l,
    ψ_z_r,
    ξ_x_l, ξ_x_r, ξ_y_l, ξ_y_r, ξ_z_l,
    ξ_z_r,
    a_x_l, a_x_r, b_K_x_l, b_K_x_r,
    a_y_l, a_y_r, b_K_y_l, b_K_y_r,
    a_z_l, a_z_r, b_K_z_l, b_K_z_r
)
    # pressure derivatives in space
    d2p_dx2 = (pcur[i+1, j, k] - 2.0 * pcur[i, j, k] + pcur[i-1, j, k]) * _dx2
    d2p_dy2 = (pcur[i, j+1, k] - 2.0 * pcur[i, j, k] + pcur[i, j-1, k]) * _dy2
    d2p_dz2 = (pcur[i, j, k+1] - 2.0 * pcur[i, j, k] + pcur[i, j, k-1]) * _dz2

    damp = 0.0
    # x boundaries
    if i <= halo + 1
        # left boundary
        dψ_x_dx = (ψ_x_l[i, j, k] - ψ_x_l[i-1, j, k]) * _dx
        ξ_x_l[i-1, j, k] = b_K_x_l[i-1] * ξ_x_l[i-1, j, k] + a_x_l[i-1] * (d2p_dx2 + dψ_x_dx)
        damp += fact[i, j, k] * (dψ_x_dx + ξ_x_l[i-1, j, k])
    elseif i >= nx - halo
        # right boundary
        ii = i - (nx - halo) + 2
        dψ_x_dx = (ψ_x_r[ii, j, k] - ψ_x_r[ii-1, j, k]) * _dx
        ξ_x_r[ii-1, j, k] = b_K_x_r[ii-1] * ξ_x_r[ii-1, j, k] + a_x_r[ii-1] * (d2p_dx2 + dψ_x_dx)
        damp += fact[i, j, k] * (dψ_x_dx + ξ_x_r[ii-1, j, k])
    end
    # y boundaries
    if j <= halo + 1
        # top boundary
        dψ_y_dy = (ψ_y_l[i, j, k] - ψ_y_l[i, j-1, k]) * _dy
        ξ_y_l[i, j-1, k] = b_K_y_l[j-1] * ξ_y_l[i, j-1, k] + a_y_l[j-1] * (d2p_dy2 + dψ_y_dy)
        damp += fact[i, j, k] * (dψ_y_dy + ξ_y_l[i, j-1, k])
    elseif j >= ny - halo
        # bottom boundary
        jj = j - (ny - halo) + 2
        dψ_y_dy = (ψ_y_r[i, jj, k] - ψ_y_r[i, jj-1, k]) * _dy
        ξ_y_r[i, jj-1, k] = b_K_y_r[jj-1] * ξ_y_r[i, jj-1, k] + a_y_r[jj-1] * (d2p_dy2 + dψ_y_dy)
        damp += fact[i, j, k] * (dψ_y_dy + ξ_y_r[i, jj-1, k])
    end
    # z boundaries
    if k <= halo + 1
        # front boundary
        dψ_z_dz = (ψ_z_l[i, j, k] - ψ_z_l[i, j, k-1]) * _dz
        ξ_z_l[i, j, k-1] = b_K_z_l[k-1] * ξ_z_l[i, j, k-1] + a_z_l[k-1] * (d2p_dz2 + dψ_z_dz)
        damp += fact[i, j, k] * (dψ_z_dz + ξ_z_l[i, j, k-1])
    elseif k >= nz - halo
        # back boundary
        kk = k - (nz - halo) + 2
        dψ_z_dz = (ψ_z_r[i, j, kk] - ψ_z_r[i, j, kk-1]) * _dz
        ξ_z_r[i, j, kk-1] = b_K_z_r[kk-1] * ξ_z_r[i, j, kk-1] + a_z_r[kk-1] * (d2p_dz2 + dψ_z_dz)
        damp += fact[i, j, k] * (dψ_z_dz + ξ_z_r[i, j, kk-1])
    end

    # update pressure
    pnew[i, j, k] =
        2.0 * pcur[i, j, k] - pold[i, j, k] + fact[i, j, k] * (d2p_dx2 + d2p_dy2 + d2p_dz2) + damp

    return nothing
end

@parallel_indices (is) function inject_sources!(pnew, dt2srctf, possrcs, it)
    isrc = floor(Int, possrcs[is, 1])
    jsrc = floor(Int, possrcs[is, 2])
    ksrc = floor(Int, possrcs[is, 3])
    pnew[isrc, jsrc, ksrc] += dt2srctf[it, is]

    return nothing
end

@parallel_indices (ir) function record_receivers!(pnew, traces, posrecs, it)
    irec = floor(Int, posrecs[ir, 1])
    jrec = floor(Int, posrecs[ir, 2])
    krec = floor(Int, posrecs[ir, 3])
    traces[it, ir] = pnew[irec, jrec, krec]

    return nothing
end

@parallel_indices (it, ir) function prescale_residuals_kernel!(residuals, posrecs, fact)
    irec = floor(Int, posrecs[ir, 1])
    jrec = floor(Int, posrecs[ir, 2])
    krec = floor(Int, posrecs[ir, 3])
    residuals[it, ir] *= fact[irec, jrec, krec]

    return nothing
end

@views function prescale_residuals!(residuals, posrecs, fact)
    nrecs = size(posrecs, 1)
    nt = size(residuals, 1)
    @parallel (1:nt, 1:nrecs) prescale_residuals_kernel!(residuals, posrecs, fact)
end

@views function forward_onestep_CPML!(
    pold, pcur, pnew, fact, dx, dy, dz, halo,
    ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r, ψ_z_l, ψ_z_r,
    ξ_x_l, ξ_x_r, ξ_y_l, ξ_y_r, ξ_z_l, ξ_z_r,
    a_x_l, a_x_r, a_x_hl, a_x_hr,
    a_y_l, a_y_r, a_y_hl, a_y_hr,
    a_z_l, a_z_r, a_z_hl, a_z_hr,
    b_K_x_l, b_K_x_r, b_K_x_hl, b_K_x_hr,
    b_K_y_l, b_K_y_r, b_K_y_hl, b_K_y_hr,
    b_K_z_l, b_K_z_r, b_K_z_hl, b_K_z_hr,
    possrcs, dt2srctf, posrecs, traces, it;
    save_trace=true
)
    nx, ny, nz = size(pcur)
    _dx = 1 / dx
    _dx2 = 1 / dx^2
    _dy = 1 / dy
    _dy2 = 1 / dy^2
    _dz = 1 / dz
    _dz2 = 1 / dz^2

    # update ψ arrays
    @parallel_async (1:(halo+1), 1:ny, 1:nz) update_ψ_x!(ψ_x_l, ψ_x_r, pcur,
        halo, _dx, nx,
        a_x_hl, a_x_hr,
        b_K_x_hl, b_K_x_hr)
    @parallel_async (1:nx, 1:(halo+1), 1:nz) update_ψ_y!(ψ_y_l, ψ_y_r, pcur,
        halo, _dy, ny,
        a_y_hl, a_y_hr,
        b_K_y_hl, b_K_y_hr)
    @parallel_async (1:nx, 1:ny, 1:(halo+1)) update_ψ_z!(ψ_z_l, ψ_z_r, pcur,
        halo, _dz, nz,
        a_z_hl, a_z_hr,
        b_K_z_hl, b_K_z_hr)
    @synchronize

    # update presure and ξ arrays
    @parallel (2:(nx-1), 2:(ny-1), 2:(nz-1)) update_p_CPML!(pold, pcur, pnew, halo,
        fact,
        _dx, _dx2, _dy, _dy2, _dz,
        _dz2, nx, ny, nz,
        ψ_x_l, ψ_x_r, ψ_y_l,
        ψ_y_r, ψ_z_l, ψ_z_r,
        ξ_x_l, ξ_x_r, ξ_y_l,
        ξ_y_r, ξ_z_l, ξ_z_r,
        a_x_l, a_x_r, b_K_x_l,
        b_K_x_r,
        a_y_l, a_y_r, b_K_y_l,
        b_K_y_r,
        a_z_l, a_z_r, b_K_z_l,
        b_K_z_r)

    # inject sources
    @parallel (1:size(possrcs, 1)) inject_sources!(pnew, dt2srctf, possrcs, it)
    # record receivers
    if save_trace
        @parallel (1:size(posrecs, 1)) record_receivers!(pnew, traces, posrecs, it)
    end

    return pcur, pnew, pold
end
