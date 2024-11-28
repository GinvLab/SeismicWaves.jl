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

@parallel_indices (i) function update_ψ!(pcur, _dx, halo, ψ_x, b_x_half, a_x_half)
    # Shift index to right side if beyond left boundary
    ii = i > halo ? size(pcur, 1) - halo - 1 + (i - halo) : i
    # Update CPML memory variable
    @∂̃x(pcur, a_x_half, b_x_half, ψ_x,
        order=2, I=(ii,), _Δ=_dx,
        halo=halo, halfgrid=true)

    return nothing
end

@parallel_indices (i) function update_p_CPML!(
    pold, pcur, pnew, fact, _dx,
    halo, ψ_x, ξ_x, b_x, a_x
)
    ##########################
    # ∂²p/∂t² = c² * ∇²p #
    # fact = c² * dt²        #
    ##########################

    # Compute pressure Laplacian
    ∇²p = @∇̃²(pcur, a_x, b_x, ψ_x, ξ_x,
              order=2, I=(i,), _Δ=(_dx,),
              halo=halo)
    # Update pressure
    pnew[i] = 2.0 * pcur[i] - pold[i] + fact[i] * ∇²p

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
    model, possrcs, dt2srctf, posrecs, traces, it;
    save_trace=true
)
    # Extract info from grid
    grid = model.grid
    nx = grid.size[1]
    dx = grid.spacing[1]
    pold, pcur, pnew = grid.fields["pold"].value, grid.fields["pcur"].value, grid.fields["pnew"].value
    fact = grid.fields["fact"].value
    ψ_x = grid.fields["ψ"].value[1]
    ξ_x = grid.fields["ξ"].value[1]
    a_x = model.cpmlcoeffs[1].a
    a_x_half = model.cpmlcoeffs[1].a_h
    b_x = model.cpmlcoeffs[1].b
    b_x_half = model.cpmlcoeffs[1].b_h
    halo = model.cpmlparams.halo
    # Precompute divisions
    _dx  = 1 / dx

    # update ψ arrays
    @parallel (1:2halo) update_ψ!(pcur, _dx, halo, ψ_x, b_x_half, a_x_half)
    # update pressure and ξ arrays
    @parallel (2:(nx-1)) update_p_CPML!(pold, pcur, pnew, fact, _dx, halo, ψ_x, ξ_x, b_x, a_x)
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

function adjoint_onestep_CPML!(model, possrcs, dt2srctf, it)
    # Extract info from grid
    grid = model.grid
    nx = grid.size[1]
    dx = grid.spacing[1]
    pold, pcur, pnew = grid.fields["adjold"].value, grid.fields["adjcur"].value, grid.fields["adjnew"].value
    fact = grid.fields["fact"].value
    ψ_x = grid.fields["ψ_adj"].value[1]
    ξ_x = grid.fields["ξ_adj"].value[1]
    a_x = model.cpmlcoeffs[1].a
    a_x_half = model.cpmlcoeffs[1].a_h
    b_x = model.cpmlcoeffs[1].b
    b_x_half = model.cpmlcoeffs[1].b_h
    halo = model.cpmlparams.halo
    # Precompute divisions
    _dx  = 1 / dx

    # update ψ arrays
    @parallel (1:2halo) update_ψ!(pcur, _dx, halo, ψ_x, b_x_half, a_x_half)
    # update pressure and ξ arrays
    @parallel (2:(nx-1)) update_p_CPML!(pold, pcur, pnew, fact, _dx, halo, ψ_x, ξ_x, b_x, a_x)
    # inject sources
    @parallel (1:size(possrcs, 1)) inject_sources!(pnew, dt2srctf, possrcs, it)

    # Exchange pressures in grid
    grid.fields["adjold"] = grid.fields["adjcur"]
    grid.fields["adjcur"] = grid.fields["adjnew"]
    grid.fields["adjnew"] = grid.fields["adjold"]
    return nothing
end
