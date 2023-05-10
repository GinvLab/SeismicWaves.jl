@parallel_indices (is) function inject_sources!(pnew, dt2srctf, possrcs, it)
    isrc = floor(Int, possrcs[is, 1])
    pnew[isrc] += dt2srctf[it, is]

    return nothing
end

@parallel_indices (ir) function record_receivers!(pnew, traces, posrecs, it)
    irec = floor(Int, posrecs[ir, 1])
    traces[it, ir] = pnew[irec]

    return nothing
end

@parallel_indices (i) function update_p!(pold, pcur, pnew, fact, _dx2)
    d2p_dx2 = (pcur[i+1] - 2.0 * pcur[i] + pcur[i-1]) * _dx2
    pnew[i] = 2.0 * pcur[i] - pold[i] + fact[i] * (d2p_dx2)

    return nothing
end

@parallel_indices (i) function update_ψ!(
    ψ_l, ψ_r, pcur,
    halo, nx, _dx,
    a_x_hl, a_x_hr,
    b_K_x_hl, b_K_x_hr
)
    ii = i + nx - halo - 2  # shift for right boundary pressure indices
    # left boundary
    ψ_l[i] = b_K_x_hl[i] * ψ_l[i] + a_x_hl[i] * (pcur[i+1] - pcur[i]) * _dx
    # right boundary
    ψ_r[i] = b_K_x_hr[i] * ψ_r[i] + a_x_hr[i] * (pcur[ii+1] - pcur[ii]) * _dx

    return nothing
end

@parallel_indices (i) function update_p_CPML!(
    pold, pcur, pnew, halo, fact, nx, _dx, _dx2,
    ψ_l, ψ_r,
    ξ_l, ξ_r,
    a_x_l, a_x_r,
    b_K_x_l, b_K_x_r
)
    d2p_dx2 = (pcur[i+1] - 2.0 * pcur[i] + pcur[i-1]) * _dx2

    if i <= halo + 1
        # left boundary
        dψ_dx = (ψ_l[i] - ψ_l[i-1]) * _dx
        ξ_l[i-1] = b_K_x_l[i-1] * ξ_l[i-1] + a_x_l[i-1] * (d2p_dx2 + dψ_dx)
        damp = fact[i] * (dψ_dx + ξ_l[i-1])
    elseif i >= nx - halo
        # right boundary
        ii = i - (nx - halo) + 2
        dψ_dx = (ψ_r[ii] - ψ_r[ii-1]) * _dx
        ξ_r[ii-1] = b_K_x_r[ii-1] * ξ_r[ii-1] + a_x_r[ii-1] * (d2p_dx2 + dψ_dx)
        damp = fact[i] * (dψ_dx + ξ_r[ii-1])
    else
        damp = 0.0
    end

    # update pressure
    pnew[i] = 2.0 * pcur[i] - pold[i] + fact[i] * (d2p_dx2) + damp

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

@views function forward_onestep!(
    pold, pcur, pnew, fact, dx,
    possrcs, dt2srctf, posrecs, traces, it;
    save_trace=true
)
    nx = length(pcur)
    _dx2 = 1 / dx^2

    @parallel (2:(nx-1)) update_p!(pold, pcur, pnew, fact, _dx2)
    @parallel (1:size(possrcs, 1)) inject_sources!(pnew, dt2srctf, possrcs, it)
    if save_trace
        @parallel (1:size(posrecs, 1)) record_receivers!(pnew, traces, posrecs, it)
    end

    return pcur, pnew, pold
end

@views function forward_onestep_CPML!(
    pold, pcur, pnew, fact, dx, halo,
    ψ_l, ψ_r, ξ_l, ξ_r,
    a_x_l, a_x_r, a_x_hl, a_x_hr,
    b_K_x_l, b_K_x_r, b_K_x_hl, b_K_x_hr,
    possrcs, dt2srctf, posrecs, traces, it;
    save_trace=true
)
    nx = length(pcur)
    _dx = 1 / dx
    _dx2 = 1 / dx^2

    @parallel (1:(halo+1)) update_ψ!(ψ_l, ψ_r, pcur,
        halo, nx, _dx,
        a_x_hl, a_x_hr,
        b_K_x_hl, b_K_x_hr)
    @parallel (2:(nx-1)) update_p_CPML!(pold, pcur, pnew, halo, fact, nx, _dx, _dx2,
        ψ_l, ψ_r,
        ξ_l, ξ_r,
        a_x_l, a_x_r,
        b_K_x_l, b_K_x_r)
    @parallel (1:size(possrcs, 1)) inject_sources!(pnew, dt2srctf, possrcs, it)
    if save_trace
        @parallel (1:size(posrecs, 1)) record_receivers!(pnew, traces, posrecs, it)
    end

    return pcur, pnew, pold
end
