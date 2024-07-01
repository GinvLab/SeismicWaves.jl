@parallel_indices (i, j) function update_ψ_x!(
    ψ_x_l, ψ_x_r, pcur,
    halo, _dx, nx,
    a_x_hl, a_x_hr,
    b_x_hl, b_x_hr
)
    ii = i + nx - halo - 2  # shift for right boundary pressure indices
    # left boundary
    ψ_x_l[i, j] = b_x_hl[i] * ψ_x_l[i, j] + a_x_hl[i] * (pcur[i+1, j] - pcur[i, j]) * _dx
    # right boundary
    ψ_x_r[i, j] = b_x_hr[i] * ψ_x_r[i, j] + a_x_hr[i] * (pcur[ii+1, j] - pcur[ii, j]) * _dx

    return nothing
end

@parallel_indices (i, j) function update_ψ_y!(
    ψ_y_l, ψ_y_r, pcur,
    halo, _dy, ny,
    a_y_hl, a_y_hr,
    b_y_hl, b_y_hr
)
    jj = j + ny - halo - 2  # shift for bottom boundary pressure indices
    # top boundary
    ψ_y_l[i, j] = b_y_hl[j] * ψ_y_l[i, j] + a_y_hl[j] * (pcur[i, j+1] - pcur[i, j]) * _dy
    # bottom boundary
    ψ_y_r[i, j] = b_y_hr[j] * ψ_y_r[i, j] + a_y_hr[j] * (pcur[i, jj+1] - pcur[i, jj]) * _dy

    return nothing
end

@parallel_indices (i, j) function update_p_CPML!(
    pold, pcur, pnew, halo, fact,
    _dx, _dx2, _dy, _dy2, nx, ny,
    ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r,
    ξ_x_l, ξ_x_r, ξ_y_l, ξ_y_r,
    a_x_l, a_x_r, b_x_l, b_x_r,
    a_y_l, a_y_r, b_y_l, b_y_r
)
    # pressure derivatives in space
    d2p_dx2 = (pcur[i+1, j] - 2.0 * pcur[i, j] + pcur[i-1, j]) * _dx2
    d2p_dy2 = (pcur[i, j+1] - 2.0 * pcur[i, j] + pcur[i, j-1]) * _dy2

    damp = 0.0
    # x boundaries
    if i <= halo + 1
        # left boundary
        dψ_x_dx = (ψ_x_l[i, j] - ψ_x_l[i-1, j]) * _dx
        ξ_x_l[i-1, j] = b_x_l[i-1] * ξ_x_l[i-1, j] + a_x_l[i-1] * (d2p_dx2 + dψ_x_dx)
        damp += fact[i, j] * (dψ_x_dx + ξ_x_l[i-1, j])
    elseif i >= nx - halo
        # right boundary
        ii = i - (nx - halo) + 2
        dψ_x_dx = (ψ_x_r[ii, j] - ψ_x_r[ii-1, j]) * _dx
        ξ_x_r[ii-1, j] = b_x_r[ii-1] * ξ_x_r[ii-1, j] + a_x_r[ii-1] * (d2p_dx2 + dψ_x_dx)
        damp += fact[i, j] * (dψ_x_dx + ξ_x_r[ii-1, j])
    end
    # y boundaries
    if j <= halo + 1
        # top boundary
        dψ_y_dy = (ψ_y_l[i, j] - ψ_y_l[i, j-1]) * _dy
        ξ_y_l[i, j-1] = b_y_l[j-1] * ξ_y_l[i, j-1] + a_y_l[j-1] * (d2p_dy2 + dψ_y_dy)
        damp += fact[i, j] * (dψ_y_dy + ξ_y_l[i, j-1])
    elseif j >= ny - halo
        # bottom boundary
        jj = j - (ny - halo) + 2
        dψ_y_dy = (ψ_y_r[i, jj] - ψ_y_r[i, jj-1]) * _dy
        ξ_y_r[i, jj-1] = b_y_r[jj-1] * ξ_y_r[i, jj-1] + a_y_r[jj-1] * (d2p_dy2 + dψ_y_dy)
        damp += fact[i, j] * (dψ_y_dy + ξ_y_r[i, jj-1])
    end

    # update pressure
    pnew[i, j] = 2.0 * pcur[i, j] - pold[i, j] + fact[i, j] * (d2p_dx2 + d2p_dy2) + damp

    return nothing
end

@parallel_indices (is) function inject_sources!(pnew, dt2srctf, possrcs, it)
    isrc = floor(Int, possrcs[is, 1])
    jsrc = floor(Int, possrcs[is, 2])
    pnew[isrc, jsrc] += dt2srctf[it, is]

    return nothing
end

@parallel_indices (ir) function record_receivers!(pnew, traces, posrecs, it)
    irec = floor(Int, posrecs[ir, 1])
    jrec = floor(Int, posrecs[ir, 2])
    traces[it, ir] = pnew[irec, jrec]

    return nothing
end

@parallel_indices (it, ir) function prescale_residuals_kernel!(residuals, posrecs, fact)
    irec = floor(Int, posrecs[ir, 1])
    jrec = floor(Int, posrecs[ir, 2])
    residuals[it, ir] *= fact[irec, jrec]

    return nothing
end

@views function prescale_residuals!(residuals, posrecs, fact)
    nrecs = size(posrecs, 1)
    nt = size(residuals, 1)
    @parallel (1:nt, 1:nrecs) prescale_residuals_kernel!(residuals, posrecs, fact)
end

@views function forward_onestep_CPML!(
    grid, possrcs, dt2srctf, posrecs, traces, it;
    save_trace=true
)
    # Extract info from grid
    nx, ny = grid.ns
    dx, dy = grid.gridspacing
    pold, pcur, pnew = grid.fields["pold"].value, grid.fields["pcur"].value, grid.fields["pnew"].value
    fact = grid.fields["fact"].value
    ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r = grid.fields["ψ"].value
    ξ_x_l, ξ_x_r, ξ_y_l, ξ_y_r = grid.fields["ξ"].value
    a_x_l, a_x_r, a_x_hl, a_x_hr, a_y_l, a_y_r, a_y_hl, a_y_hr = grid.fields["a_pml"].value
    b_x_l, b_x_r, b_x_hl, b_x_hr, b_y_l, b_y_r, b_y_hl, b_y_hr = grid.fields["b_pml"].value
    halo = length(a_x_r)
    _dx = 1 / dx
    _dx2 = 1 / dx^2
    _dy = 1 / dy
    _dy2 = 1 / dy^2

    # update ψ arrays
    @parallel_async (1:(halo+1), 1:ny) update_ψ_x!(ψ_x_l, ψ_x_r, pcur,
        halo, _dx, nx,
        a_x_hl, a_x_hr,
        b_x_hl, b_x_hr)
    @parallel_async (1:nx, 1:(halo+1)) update_ψ_y!(ψ_y_l, ψ_y_r, pcur,
        halo, _dy, ny,
        a_y_hl, a_y_hr,
        b_y_hl, b_y_hr)
    @synchronize

    # update pressure and ξ arrays
    @parallel (2:(nx-1), 2:(ny-1)) update_p_CPML!(pold, pcur, pnew, halo, fact,
        _dx, _dx2, _dy, _dy2, nx, ny,
        ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r,
        ξ_x_l, ξ_x_r, ξ_y_l, ξ_y_r,
        a_x_l, a_x_r, b_x_l, b_x_r,
        a_y_l, a_y_r, b_y_l, b_y_r)

    # inject sources
    @parallel (1:size(possrcs, 1)) inject_sources!(pnew, dt2srctf, possrcs, it)
    # record receivers
    if save_trace
        @parallel (1:size(posrecs, 1)) record_receivers!(pnew, traces, posrecs, it)
    end

    # Exchange pressures in grid
    grid.fields["pold"] = grid.fields["pcur"]
    grid.fields["pcur"] = grid.fields["pnew"]
    grid.fields["pnew"] = grid.fields["pold"]
    return nothing
end

@views function adjoint_onestep_CPML!(
    grid, possrcs, dt2srctf, posrecs, traces, it;
    save_trace=true
)
    # Extract info from grid
    nx, ny = grid.ns
    dx, dy = grid.gridspacing
    pold, pcur, pnew = grid.fields["adjold"].value, grid.fields["adjcur"].value, grid.fields["adjnew"].value
    fact = grid.fields["fact"].value
    ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r = grid.fields["ψ_adj"].value
    ξ_x_l, ξ_x_r, ξ_y_l, ξ_y_r = grid.fields["ξ_adj"].value
    a_x_l, a_x_r, a_x_hl, a_x_hr, a_y_l, a_y_r, a_y_hl, a_y_hr = grid.fields["a_pml"].value
    b_x_l, b_x_r, b_x_hl, b_x_hr, b_y_l, b_y_r, b_y_hl, b_y_hr = grid.fields["b_pml"].value
    halo = length(a_x_r)
    # Precompute divisions
    _dx = 1 / dx
    _dx2 = 1 / dx^2
    _dy = 1 / dy
    _dy2 = 1 / dy^2

    # update ψ arrays
    @parallel_async (1:(halo+1), 1:ny) update_ψ_x!(ψ_x_l, ψ_x_r, pcur,
        halo, _dx, nx,
        a_x_hl, a_x_hr,
        b_x_hl, b_x_hr)
    @parallel_async (1:nx, 1:(halo+1)) update_ψ_y!(ψ_y_l, ψ_y_r, pcur,
        halo, _dy, ny,
        a_y_hl, a_y_hr,
        b_y_hl, b_y_hr)
    @synchronize

    # update pressure and ξ arrays
    @parallel (2:(nx-1), 2:(ny-1)) update_p_CPML!(pold, pcur, pnew, halo, fact,
        _dx, _dx2, _dy, _dy2, nx, ny,
        ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r,
        ξ_x_l, ξ_x_r, ξ_y_l, ξ_y_r,
        a_x_l, a_x_r, b_x_l, b_x_r,
        a_y_l, a_y_r, b_y_l, b_y_r)

    # inject sources
    @parallel (1:size(possrcs, 1)) inject_sources!(pnew, dt2srctf, possrcs, it)
    # record receivers
    if save_trace
        @parallel (1:size(posrecs, 1)) record_receivers!(pnew, traces, posrecs, it)
    end

    # Exchange pressures in grid
    grid.fields["adjold"] = grid.fields["adjcur"]
    grid.fields["adjcur"] = grid.fields["adjnew"]
    grid.fields["adjnew"] = grid.fields["adjold"]
    return nothing
end
