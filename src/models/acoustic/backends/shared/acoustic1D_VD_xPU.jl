@parallel_indices (is) function inject_sources!(pcur, srctf, possrcs, it)
    isrc = floor(Int, possrcs[is, 1])
    pcur[isrc] += srctf[it, is]

    return nothing
end

@parallel_indices (ir) function record_receivers!(pcur, traces, posrecs, it)
    irec = floor(Int, posrecs[ir, 1])
    traces[it, ir] = pcur[irec]

    return nothing
end

@parallel_indices (i) function update_p_CPML!(pcur, vx_cur, halo, fact_m0, nx, _dx,
    ξ_l, ξ_r, a_x_l, a_x_r, b_x_l, b_x_r)
    # Compute velocity derivatives
    dv_dx = @d_dx_4th(vx_cur, i) * _dx
    # Update CPML memory arrays if on the boundary
    if i <= halo + 1
        # left boundary
        ξ_l[i-1] = b_x_l[i-1] * ξ_l[i-1] + a_x_l[i-1] * dv_dx
        dv_dx += ξ_l[i-1]
    elseif i >= nx - halo
        # right boundary
        ii = i - (nx - halo) + 1
        ξ_r[ii] = b_x_r[ii] * ξ_r[ii] + a_x_r[ii] * dv_dx
        dv_dx += ξ_r[ii]
    end
    # Update pressure
    pcur[i] -= fact_m0[i] * dv_dx

    return nothing
end

@parallel_indices (i) function update_vx_CPML!(pcur, vx_cur, halo, fact_m1_x, nx, _dx,
    ψ_l, ψ_r, a_x_hl, a_x_hr, b_x_hl, b_x_hr)
    # Compute pressure derivative in x direction
    dp_dx = @d_dx_4th(pcur, i + 1) * _dx
    # Update CPML memory arrays if on the boundary
    if i <= halo + 1
        # left boundary
        ψ_l[i] = b_x_hl[i] * ψ_l[i] + a_x_hl[i] * dp_dx
        dp_dx += ψ_l[i]
    elseif i >= nx - halo - 1
        # right boundary
        ii = i - (nx - halo - 1) + 1
        ψ_r[ii] = b_x_hr[ii] * ψ_r[ii] + a_x_hr[ii] * dp_dx
        dp_dx += ψ_r[ii]
    end
    # Update velocity
    vx_cur[i] -= fact_m1_x[i] * dp_dx

    return nothing
end

@parallel_indices (it, ir) function prescale_residuals_kernel!(residuals, posrecs, fact)
    irec = floor(Int, posrecs[ir, 1])
    residuals[it, ir] *= fact[irec]

    return nothing
end

@views function prescale_residuals!(residuals, posrecs, fact)
    nrecs = size(posrecs, 1)
    nt = size(residuals, 1)
    @parallel (1:nt, 1:nrecs) prescale_residuals_kernel!(residuals, posrecs, fact)
end

@views function forward_onestep_CPML!(
    grid, possrcs, srctf, posrecs, traces, it;
    save_trace=true
)
    # Extract info from grid
    nx = grid.ns[1]
    dx = grid.gridspacing[1]
    pcur, vx_cur = grid.fields["pcur"].value, grid.fields["vcur"].value[1]
    fact_m0 = grid.fields["fact_m0"].value
    fact_m1_x = grid.fields["fact_m1_stag"].value[1]
    ψ_l, ψ_r = grid.fields["ψ"].value
    ξ_l, ξ_r = grid.fields["ξ"].value
    a_x_l, a_x_r, a_x_hl, a_x_hr = grid.fields["a_pml"].value
    b_x_l, b_x_r, b_x_hl, b_x_hr = grid.fields["b_pml"].value
    halo = length(a_x_r)
    # Precompute divisions
    _dx = 1 / (dx * 24)

    @parallel (3:(nx-2)) update_p_CPML!(pcur, vx_cur, halo, fact_m0, nx, _dx,
        ξ_l, ξ_r, a_x_l, a_x_r, b_x_l, b_x_r)
    @parallel (1:size(possrcs, 1)) inject_sources!(pcur, srctf, possrcs, it)
    @parallel (2:(nx-2)) update_vx_CPML!(pcur, vx_cur, halo, fact_m1_x, nx, _dx,
        ψ_l, ψ_r, a_x_hl, a_x_hr, b_x_hl, b_x_hr)
    if save_trace
        @parallel (1:size(posrecs, 1)) record_receivers!(pcur, traces, posrecs, it)
    end
end

@views function adjoint_onestep_CPML!(
    grid, possrcs, srctf, it
)
    # Extract info from grid
    nx = grid.ns[1]
    dx = grid.gridspacing[1]
    pcur, vx_cur = grid.fields["adjpcur"].value, grid.fields["adjvcur"].value[1]
    fact_m0 = grid.fields["fact_m0"].value
    fact_m1_x = grid.fields["fact_m1_stag"].value[1]
    ψ_l, ψ_r = grid.fields["ψ_adj"].value
    ξ_l, ξ_r = grid.fields["ξ_adj"].value
    a_x_l, a_x_r, a_x_hl, a_x_hr = grid.fields["a_pml"].value
    b_x_l, b_x_r, b_x_hl, b_x_hr = grid.fields["b_pml"].value
    halo = length(a_x_r)
    # Precompute divisions
    _dx = 1 / (dx * 24)

    @parallel (2:(nx-2)) update_vx_CPML!(pcur, vx_cur, halo, fact_m1_x, nx, _dx,
        ψ_l, ψ_r, a_x_hl, a_x_hr, b_x_hl, b_x_hr)
    @parallel (3:(nx-2)) update_p_CPML!(pcur, vx_cur, halo, fact_m0, nx, _dx,
        ξ_l, ξ_r, a_x_l, a_x_r, b_x_l, b_x_r)
    @parallel (1:size(possrcs, 1)) inject_sources!(pcur, srctf, possrcs, it)
end

@views function correlate_gradient_m1!(curgrad_m1_stag, adjvcur, pold, gridspacing)
    _gridspacing = 1 ./ (gridspacing .* 24)
    nx = length(pold)
    @parallel (2:nx-2) correlate_gradient_m1_kernel_x!(curgrad_m1_stag[1], adjvcur[1], pold, _gridspacing[1])
end

@parallel_indices (i) function correlate_gradient_m1_kernel_x!(curgrad_m1_stag_x, adjvcur_x, pold, _dx)
    curgrad_m1_stag_x[i] = curgrad_m1_stag_x[i] + adjvcur_x[i] * @d_dx_4th(pold, i + 1) * _dx

    return nothing
end