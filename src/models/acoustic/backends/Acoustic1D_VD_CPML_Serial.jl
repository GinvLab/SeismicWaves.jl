module Acoustic1D_VD_CPML_Serial

# Dummy data module
module Data
Array = Base.Array
end

#####
ones = Base.ones
zeros = Base.zeros

@views function inject_sources!(pnew, dt2srctf, possrcs, it)
    _, nsrcs = size(dt2srctf)
    for i in 1:nsrcs
        isrc = possrcs[i, 1]
        pnew[isrc] += dt2srctf[it, i]
    end
end

@views function record_receivers!(pnew, traces, posrecs, it)
    _, nrecs = size(traces)
    for s in 1:nrecs
        irec = posrecs[s, 1]
        traces[it, s] = pnew[irec]
    end
end

@views update_p!(pold, pcur, pnew, fact_vp2rho, fact_rho_stag_x, nx, _dx) =
    for i in 2:nx-1
        d2p_dx2 = ((pcur[i+1] - pcur[i]) * _dx * fact_rho_stag_x[i] - (pcur[i] - pcur[i-1]) * _dx * fact_rho_stag_x[i-1]) * _dx
        pnew[i] = 2 * pcur[i] - pold[i] + fact_vp2rho[i] * d2p_dx2
    end

@views function update_ψ!(ψ_l, ψ_r, pcur, halo, nx, _dx, a_x_hl, a_x_hr, b_x_hl, b_x_hr)
    for i in 1:(halo+1)
        ii = i + nx - halo - 2  # shift for right boundary pressure indices
        # left boundary
        ψ_l[i] = b_x_hl[i] * ψ_l[i] + a_x_hl[i] * (pcur[i+1] - pcur[i]) * _dx
        # right boundary
        ψ_r[i] = b_x_hr[i] * ψ_r[i] + a_x_hr[i] * (pcur[ii+1] - pcur[ii]) * _dx
    end
end

@views function update_p_CPML!(
    pold, pcur, pnew, halo, fact_vp2rho, fact_rho_stag_x, nx, _dx,
    ψ_l, ψ_r,
    ξ_l, ξ_r,
    a_x_l, a_x_r,
    b_x_l, b_x_r
)
    for i in 2:(nx-1)
        d2p_dx2 = ((pcur[i+1] - pcur[i]) * _dx * fact_rho_stag_x[i] - (pcur[i] - pcur[i-1]) * _dx * fact_rho_stag_x[i-1]) * _dx

        if i <= halo + 1
            # left boundary
            dψ_dx = (ψ_l[i] * fact_rho_stag_x[i] - ψ_l[i-1] * fact_rho_stag_x[i-1]) * _dx
            ξ_l[i-1] = b_x_l[i-1] * ξ_l[i-1] + a_x_l[i-1] * (d2p_dx2 + dψ_dx)
            damp = fact_vp2rho[i] * (dψ_dx + ξ_l[i-1])
        elseif i >= nx - halo
            # right boundary
            ii = i - (nx - halo) + 2
            dψ_dx = (ψ_r[ii] * fact_rho_stag_x[ii] - ψ_r[ii-1] * fact_rho_stag_x[ii-1]) * _dx
            ξ_r[ii-1] = b_x_r[ii-1] * ξ_r[ii-1] + a_x_r[ii-1] * (d2p_dx2 + dψ_dx)
            damp = fact_vp2rho[i] * (dψ_dx + ξ_r[ii-1])
        else
            damp = 0.0
        end

        # update pressure
        pnew[i] = 2.0 * pcur[i] - pold[i] + fact_vp2rho[i] * (d2p_dx2) + damp
    end
end

@views prescale_residuals!(residuals, possrcs, fact) =
    for is in axes(possrcs, 1)
        isrc = floor(Int, possrcs[is, 1])
        for it in axes(residuals, 1) # nt
            residuals[it, is] *= fact[isrc]
        end
    end

@views function forward_onestep!(
    pold, pcur, pnew, fact_vp2rho, fact_rho_stag_x, dx,
    possrcs, dt2srctf, posrecs, traces, it
)
    nx = length(pcur)
    _dx = 1 / dx

    update_p!(pold, pcur, pnew, fact_vp2rho, fact_rho_stag_x, nx, _dx)
    inject_sources!(pnew, dt2srctf, possrcs, it)
    record_receivers!(pnew, traces, posrecs, it)

    return pcur, pnew, pold
end

@views function forward_onestep_CPML!(
    pold, pcur, pnew, fact_vp2rho, fact_rho_stag_x, dx,
    halo, ψ_l, ψ_r, ξ_l, ξ_r,
    a_x_l, a_x_r, a_x_hl, a_x_hr,
    b_x_l, b_x_r, b_x_hl, b_x_hr,
    possrcs, dt2srctf, posrecs, traces, it;
    save_trace=true
)
    nx = length(pcur)
    _dx = 1 / dx

    update_ψ!(ψ_l, ψ_r, pcur,
        halo, nx, _dx,
        a_x_hl, a_x_hr,
        b_x_hl, b_x_hr)
    update_p_CPML!(pold, pcur, pnew, halo, fact_vp2rho, fact_rho_stag_x, nx, _dx,
        ψ_l, ψ_r,
        ξ_l, ξ_r,
        a_x_l, a_x_r,
        b_x_l, b_x_r)
    inject_sources!(pnew, dt2srctf, possrcs, it)
    if save_trace
        record_receivers!(pnew, traces, posrecs, it)
    end

    return pcur, pnew, pold
end

function correlate_gradient_vp!(curgrad, adjcur, pnew, pcur, pold, dt)
    _dt2 = 1 / dt^2
    nx = size(curgrad, 1)
    for i in 1:nx
        curgrad[i] = curgrad[i] + (adjcur[i] * (pnew[i] - 2.0 * pcur[i] + pold[i]) * _dt2)
    end
end

function correlate_gradient_rho!(curgrad, adjcur, pnew, pcur, pold,
    fact_c2_dt2, fact_rho2, fact_rhostag2_x, fact_dh_drhostag_x,
    possrcs, dt2srctf, dx, it)
    _dx = 1 / dx
    nx = size(curgrad, 1)
    for i in 2:nx-1
        # Compute d2p_dt2 contributions
        d2p_dt2_c2 = (pnew[i] - 2.0 * pcur[i] + pold[i]) * fact_c2_dt2[i]
        curgrad[i] = curgrad[i] - d2p_dt2_c2 * fact_rho2[i]
        # Compute grad(p)*grad(adj) contributions
        right = (pcur[i+1] - pcur[i]) * _dx * (adjcur[i+1] - adjcur[i]) * _dx * fact_rhostag2_x[i] * fact_dh_drhostag_x[2]
        left = (pcur[i] - pcur[i-1]) * _dx * (adjcur[i] - adjcur[i-1]) * _dx * fact_rhostag2_x[i-1] * fact_dh_drhostag_x[1]
        curgrad[i] = curgrad[i] - (right + left)
    end
    # Compute sources contributions
    _, nsrcs = size(dt2srctf)
    for i in 1:nsrcs
        isrc = possrcs[i, 1]
        curgrad[isrc] = curgrad[isrc] + dt2srctf[it, i] * fact_c2_dt2[isrc] * fact_rho2[isrc]
    end
end

###################
end # module
###################
