module Acoustic1D_CD_CPML_Serial

include("shared/smooth_gradient_1D.jl")

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
        isrc = floor(Int, possrcs[i, 1])
        pnew[isrc] += dt2srctf[it, i]
    end
end

@views function record_receivers!(pnew, traces, posrecs, it)
    _, nrecs = size(traces)
    for s in 1:nrecs
        irec = floor(Int, posrecs[s, 1])
        traces[it, s] = pnew[irec]
    end
end

@views update_p!(pold, pcur, pnew, fact, nx, _dx2) =
    for i in 2:(nx-1)
        d2p_dx2 = (pcur[i-1] - 2.0 * pcur[i] + pcur[i-1]) * _dx2
        pnew[i] = 2.0 * pcur[i] - pold[i] + fact[i] * (d2p_dx2)
    end

@views update_ψ!(ψ_l, ψ_r, pcur, halo, nx, _dx, a_x_hl, a_x_hr, b_x_hl, b_x_hr) =
    for i in 1:(halo+1)
        ii = i + nx - halo - 2  # shift for right boundary pressure indices
        # left boundary
        ψ_l[i] = b_x_hl[i] * ψ_l[i] + a_x_hl[i] * (pcur[i+1] - pcur[i]) * _dx
        # right boundary
        ψ_r[i] = b_x_hr[i] * ψ_r[i] + a_x_hr[i] * (pcur[ii+1] - pcur[ii]) * _dx
    end

@views function update_p_CPML!(
    pold, pcur, pnew, halo, fact, nx, _dx, _dx2,
    ψ_l, ψ_r,
    ξ_l, ξ_r,
    a_x_l, a_x_r,
    b_x_l, b_x_r
)
    for i in 2:(nx-1)
        d2p_dx2 = (pcur[i+1] - 2.0 * pcur[i] + pcur[i-1]) * _dx2

        if i <= halo + 1
            # left boundary
            dψ_dx = (ψ_l[i] - ψ_l[i-1]) * _dx
            ξ_l[i-1] = b_x_l[i-1] * ξ_l[i-1] + a_x_l[i-1] * (d2p_dx2 + dψ_dx)
            damp = fact[i] * (dψ_dx + ξ_l[i-1])
        elseif i >= nx - halo
            # right boundary
            ii = i - (nx - halo) + 2
            dψ_dx = (ψ_r[ii] - ψ_r[ii-1]) * _dx
            ξ_r[ii-1] = b_x_r[ii-1] * ξ_r[ii-1] + a_x_r[ii-1] * (d2p_dx2 + dψ_dx)
            damp = fact[i] * (dψ_dx + ξ_r[ii-1])
        else
            damp = 0.0
        end

        # update pressure
        pnew[i] = 2.0 * pcur[i] - pold[i] + fact[i] * (d2p_dx2) + damp
    end
end

@views prescale_residuals!(residuals, posrecs, fact) =
    for ir in axes(posrecs, 1)
        irec = floor(Int, posrecs[ir, 1])
        for it in axes(residuals, 1) # nt
            residuals[it, ir] *= fact[irec]
        end
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
    grid, possrcs, dt2srctf, posrecs, traces, it;
    save_trace=true
)
    # Extract info from grid
    nx = grid.ns[1]
    dx = grid.gridspacing[1]
    pold, pcur, pnew = grid.fields["pold"].value, grid.fields["pcur"].value, grid.fields["pnew"].value
    fact = grid.fields["fact"].value
    ψ_l, ψ_r = grid.fields["ψ"].value
    ξ_l, ξ_r = grid.fields["ξ"].value
    a_x_l, a_x_r, a_x_hl, a_x_hr = grid.fields["a_pml"].value
    b_x_l, b_x_r, b_x_hl, b_x_hr = grid.fields["b_pml"].value
    halo = length(a_x_r)
    # Precompute divisions
    _dx = 1 / dx
    _dx2 = 1 / dx^2

    update_ψ!(ψ_l, ψ_r, pcur,
        halo, nx, _dx,
        a_x_hl, a_x_hr,
        b_x_hl, b_x_hr)
    update_p_CPML!(pold, pcur, pnew, halo, fact, nx, _dx, _dx2,
        ψ_l, ψ_r,
        ξ_l, ξ_r,
        a_x_l, a_x_r,
        b_x_l, b_x_r)
    inject_sources!(pnew, dt2srctf, possrcs, it)
    if save_trace
        record_receivers!(pnew, traces, posrecs, it)
    end

    # Exchange pressures in grid
    grid.fields["pold"] = grid.fields["pcur"]
    grid.fields["pcur"] = grid.fields["pnew"]
    grid.fields["pnew"] = grid.fields["pold"]

    return nothing
end

@views function adjoint_onestep_CPML!(grid, possrcs, dt2srctf, it)
    # Extract info from grid
    nx = grid.ns[1]
    dx = grid.gridspacing[1]
    pold, pcur, pnew = grid.fields["adjold"].value, grid.fields["adjcur"].value, grid.fields["adjnew"].value
    fact = grid.fields["fact"].value
    ψ_l, ψ_r = grid.fields["ψ_adj"].value
    ξ_l, ξ_r = grid.fields["ξ_adj"].value
    a_x_l, a_x_r, a_x_hl, a_x_hr = grid.fields["a_pml"].value
    b_x_l, b_x_r, b_x_hl, b_x_hr = grid.fields["b_pml"].value
    halo = length(a_x_r)
    # Precompute divisions
    _dx = 1 / dx
    _dx2 = 1 / dx^2

    update_ψ!(ψ_l, ψ_r, pcur,
        halo, nx, _dx,
        a_x_hl, a_x_hr,
        b_x_hl, b_x_hr)
    update_p_CPML!(pold, pcur, pnew, halo, fact, nx, _dx, _dx2,
        ψ_l, ψ_r,
        ξ_l, ξ_r,
        a_x_l, a_x_r,
        b_x_l, b_x_r)
    inject_sources!(pnew, dt2srctf, possrcs, it)

    # Exchange pressures in grid
    grid.fields["adjold"] = grid.fields["adjcur"]
    grid.fields["adjcur"] = grid.fields["adjnew"]
    grid.fields["adjnew"] = grid.fields["adjold"]

    return nothing
end

function correlate_gradient!(curgrad, adjcur, pcur, pold, pveryold, dt)
    _dt2 = 1 / dt^2
    nx = size(curgrad, 1)
    for i in 1:nx
        curgrad[i] = curgrad[i] + (adjcur[i] * (pcur[i] - 2.0 * pold[i] + pveryold[i]) * _dt2)
    end
end

###################
end # module
###################
