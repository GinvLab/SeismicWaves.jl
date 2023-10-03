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
    dv_dx = (vx_cur[i] - vx_cur[i-1]) * _dx
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
    dp_dx = (pcur[i+1] - pcur[i]) * _dx
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
    pcur, vx_cur, fact_m0, fact_m1_x, dx, halo,
    ψ_l, ψ_r, ξ_l, ξ_r,
    a_x_l, a_x_r, a_x_hl, a_x_hr,
    b_x_l, b_x_r, b_x_hl, b_x_hr,
    possrcs, srctf, posrecs, traces, it;
    save_trace=true
)
    nx = length(pcur)
    _dx = 1 / dx

    @parallel (2:(nx-1)) update_p_CPML!(pcur, vx_cur, halo, fact_m0, nx, _dx,
                                        ξ_l, ξ_r, a_x_l, a_x_r, b_x_l, b_x_r)
    @parallel (1:size(possrcs, 1)) inject_sources!(pcur, srctf, possrcs, it)
    @parallel (1:(nx-1)) update_vx_CPML!(pcur, vx_cur, halo, fact_m1_x, nx, _dx,
                                         ψ_l, ψ_r, a_x_hl, a_x_hr, b_x_hl, b_x_hr)
    if save_trace
        @parallel (1:size(posrecs, 1)) record_receivers!(pcur, traces, posrecs, it)
    end

end

@views function backward_onestep_CPML!(
    pcur, vx_cur, fact_m0, fact_m1_x, dx, halo,
    ψ_l, ψ_r, ξ_l, ξ_r,
    a_x_l, a_x_r, a_x_hl, a_x_hr,
    b_x_l, b_x_r, b_x_hl, b_x_hr,
    possrcs, srctf, it
)
    nx = length(pcur)
    _dx = 1 / dx

    @parallel (1:(nx-1)) update_vx_CPML!(pcur, vx_cur, halo, fact_m1_x, nx, _dx,
                                         ψ_l, ψ_r, a_x_hl, a_x_hr, b_x_hl, b_x_hr)
    @parallel (2:(nx-1)) update_p_CPML!(pcur, vx_cur, halo, fact_m0, nx, _dx,
                                        ξ_l, ξ_r, a_x_l, a_x_r, b_x_l, b_x_r)
    @parallel (1:size(possrcs, 1)) inject_sources!(pcur, srctf, possrcs, it)
end

@parallel function correlate_gradient_m1_kernel!(curgrad_m1_stag_x, adjvcur_x, pold, _dx)
    @all(curgrad_m1_stag_x) = @all(curgrad_m1_stag_x) + (@all(adjvcur_x) * @d(pold) * _dx)

    return nothing
end