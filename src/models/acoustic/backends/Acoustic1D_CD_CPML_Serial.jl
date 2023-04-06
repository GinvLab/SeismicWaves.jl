module Acoustic1D_CD_CPML_Serial

# Dummy data module
module Data
    Array = Base.Array
end

#####
ones = Base.ones
zeros = Base.zeros


@views function inject_sources!(pnew, dt2srctf, possrcs, it)
    _, nsrcs = size(dt2srctf)
    for i = 1:nsrcs
        isrc = possrcs[i,1]
        pnew[isrc] += dt2srctf[it,i]
    end
end

@views function record_receivers!(pnew, traces, posrecs, it)
    _, nrecs = size(traces)
    for s = 1:nrecs
        irec = posrecs[s,1]
        traces[it,s] = pnew[irec]
    end
end

@views function update_p!(pold, pcur, pnew, fact, nx, _dx2)
    for i = 2:nx-1
        d2p_dx2 = (pcur[i+1] - 2.0*pcur[i] + pcur[i-1])*_dx2
        pnew[i] = 2.0 * pcur[i] - pold[i] + fact[i] * (d2p_dx2)
    end
end

@views function update_ψ!(ψ_l, ψ_r, pcur,
                          halo, nx, _dx,
                          a_x_hl, a_x_hr,
                          b_K_x_hl, b_K_x_hr)
    for i = 1:halo+1
        ii = i + nx - halo - 2  # shift for right boundary pressure indices
        # left boundary
        ψ_l[i] = b_K_x_hl[i] * ψ_l[i] + a_x_hl[i]*(pcur[ i+1] - pcur[ i])*_dx
        # right boundary
        ψ_r[i] = b_K_x_hr[i] * ψ_r[i] + a_x_hr[i]*(pcur[ii+1] - pcur[ii])*_dx
    end
    return
end

@views function update_p_CPML!(pold, pcur, pnew, halo, fact, nx, _dx, _dx2,
                               ψ_l, ψ_r,
                               ξ_l, ξ_r,
                               a_x_l, a_x_r,
                               b_K_x_l, b_K_x_r)
    for i = 2:nx-1
        d2p_dx2 = (pcur[i+1] - 2.0*pcur[i] + pcur[i-1])*_dx2

        if i <= halo+1
            # left boundary
            dψ_dx = (ψ_l[i] - ψ_l[i-1])*_dx
            ξ_l[i-1] = b_K_x_l[i-1] * ξ_l[i-1] + a_x_l[i-1] * (d2p_dx2 + dψ_dx)
            damp = fact[i] * (dψ_dx + ξ_l[i-1])
        elseif i >= nx - halo
            # right boundary
            ii = i - (nx - halo) + 2
            dψ_dx = (ψ_r[ii] - ψ_r[ii-1])*_dx
            ξ_r[ii-1] = b_K_x_r[ii-1] * ξ_r[ii-1] + a_x_r[ii-1] * (d2p_dx2 + dψ_dx)
            damp = fact[i] * (dψ_dx + ξ_r[ii-1])
        else
            damp = 0.0
        end

        # update pressure
        pnew[i] = 2.0 * pcur[i] - pold[i] + fact[i] * (d2p_dx2) + damp
    end
    return
end

@views function prescale_residuals!(residuals, possrcs, fact)
    for is = axes(possrcs,1)
        isrc = floor(Int, possrcs[is,1])
        for it= axes(residuals,1) # nt
            residuals[it,is] *= fact[isrc]
        end
    end
    return nothing
end


@views function forward_onestep!(
    pold, pcur, pnew, fact, dx,
    possrcs, dt2srctf, posrecs, traces, it
)
    nx = length(pcur)
    _dx2 = 1 / dx^2

    update_p!(pold, pcur, pnew, fact, nx, _dx2)
    inject_sources!(pnew, dt2srctf, possrcs, it)
    record_receivers!(pnew, traces, posrecs, it)

    return pcur, pnew, pold
end

@views function forward_onestep_CPML!(
    pold, pcur, pnew, fact, dx,
    halo, ψ_l, ψ_r, ξ_l, ξ_r,
    a_x_l, a_x_r, a_x_hl, a_x_hr,
    b_K_x_l, b_K_x_r,b_K_x_hl, b_K_x_hr,
    possrcs, dt2srctf, posrecs, traces, it;
    save_trace=true
)
    nx = length(pcur)
    _dx = 1 / dx
    _dx2 = 1 / dx^2

    update_ψ!(
        ψ_l, ψ_r, pcur,
        halo, nx, _dx,
        a_x_hl, a_x_hr,
        b_K_x_hl, b_K_x_hr
    )
    update_p_CPML!(
        pold, pcur, pnew, halo, fact, nx, _dx, _dx2,
        ψ_l, ψ_r,
        ξ_l, ξ_r,
        a_x_l, a_x_r,
        b_K_x_l, b_K_x_r
    )
    inject_sources!(pnew, dt2srctf, possrcs, it)
    if save_trace
        record_receivers!(pnew, traces, posrecs, it)
    end

    return pcur, pnew, pold
end


function correlate_gradient!(curgrad, adjcur, pcur, pold, pveryold, dt)
    _dt2 = 1/dt^2
    nx = size(curgrad,1)
    for i=1:nx
        curgrad[i] = curgrad[i] + ( adjcur[i] * ( pcur[i] - 2.0 * pold[i] + pveryold[i] ) * _dt2 )
    end
    return nothing
end





###################
end # module
###################
