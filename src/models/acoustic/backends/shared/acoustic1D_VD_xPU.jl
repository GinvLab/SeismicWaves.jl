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

@parallel_indices (i) function update_p_CPML!(
    pcur, vx_cur, fact_m0, _dx,
    halo, ξ_x, b_x, a_x
)
    #########################
    # ∂p/∂t = -∂vx/∂x / m0  #
    # fact_m0 = 1 / m0 / dt #
    #########################

    # Compute velocity derivative
    ∂vx∂x = @∂̃x(vx_cur, a_x, b_x, ξ_x,
                order=4, I=(i-1,), _Δ=_dx,
                halo=halo, halfgrid=false)
    # Update pressure
    pcur[i] -= fact_m0[i] * ∂vx∂x

    return nothing
end

@parallel_indices (i) function update_vx_CPML!(
    pcur, vx_cur, fact_m1_x, _dx,
    halo, ψ_x, b_x_half, a_x_half
)
    ############################
    # ∂vx/∂t = -∂p/∂x * m1     #
    # fact_m1_x = m1 / dt      #
    ############################

    # Compute pressure derivative
    ∂p∂x = @∂̃x(pcur, a_x_half, b_x_half, ψ_x,
               order=4, I=(i,), _Δ=_dx,
               halo=halo, halfgrid=true)
    # Update velocity
    vx_cur[i] -= fact_m1_x[i] * ∂p∂x

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
    model, possrcs, srctf, posrecs, traces, it;
    save_trace=true
)
    # Extract info from grid
    grid = model.grid
    nx = grid.size[1]
    dx = grid.spacing[1]
    pcur, vx_cur = grid.fields["pcur"].value, grid.fields["vcur"].value[1]
    fact_m0 = grid.fields["fact_m0"].value
    fact_m1_x = grid.fields["fact_m1_stag"].value[1]
    ψ_x = grid.fields["ψ"].value[1]
    ξ_x = grid.fields["ξ"].value[1]
    a_x = model.cpmlcoeffs[1].a
    a_x_half = model.cpmlcoeffs[1].a_h
    b_x = model.cpmlcoeffs[1].b
    b_x_half = model.cpmlcoeffs[1].b_h
    halo = model.cpmlparams.halo
    # Precompute divisions
    _dx = 1 / dx

    @parallel (2:(nx-1)) update_p_CPML!(
        pcur, vx_cur, fact_m0, _dx,
        halo, ξ_x, b_x, a_x
    )
    @parallel (1:size(possrcs, 1)) inject_sources!(pcur, srctf, possrcs, it)
    @parallel (1:(nx-1)) update_vx_CPML!(
        pcur, vx_cur, fact_m1_x, _dx,
        halo, ψ_x, b_x_half, a_x_half
    )
    if save_trace
        @parallel (1:size(posrecs, 1)) record_receivers!(pcur, traces, posrecs, it)
    end
end

@views function adjoint_onestep_CPML!(
    model, possrcs, srctf, it
)
    # Extract info from grid
    grid = model.grid
    nx = grid.size[1]
    dx = grid.spacing[1]
    pcur, vx_cur = grid.fields["adjpcur"].value, grid.fields["adjvcur"].value[1]
    fact_m0 = grid.fields["fact_m0"].value
    fact_m1_x = grid.fields["fact_m1_stag"].value[1]
    ψ_x = grid.fields["ψ_adj"].value[1]
    ξ_x = grid.fields["ξ_adj"].value[1]
    a_x = model.cpmlcoeffs[1].a
    a_x_half = model.cpmlcoeffs[1].a_h
    b_x = model.cpmlcoeffs[1].b
    b_x_half = model.cpmlcoeffs[1].b_h
    halo = model.cpmlparams.halo
    # Precompute divisions
    _dx = 1 / dx

    @parallel (1:(nx-1)) update_vx_CPML!(
        pcur, vx_cur, fact_m1_x, _dx,
        halo, ψ_x, b_x_half, a_x_half
    )
    @parallel (2:(nx-1)) update_p_CPML!(
        pcur, vx_cur, fact_m0, _dx,
        halo, ξ_x, b_x, a_x
    )
    @parallel (1:size(possrcs, 1)) inject_sources!(pcur, srctf, possrcs, it)
end

@views function correlate_gradient_m1!(curgrad_m1_stag, adjvcur, pold, gridspacing)
    _gridspacing = 1 ./ gridspacing
    nx = length(pold)
    @parallel (2:nx-2) correlate_gradient_m1_kernel_x!(curgrad_m1_stag[1], adjvcur[1], pold, _gridspacing[1])
end

@parallel_indices (i) function correlate_gradient_m1_kernel_x!(curgrad_m1_stag_x, adjvcur_x, pold, _dx)
    ∂p∂x = @∂x(pold, order=4, I=(i,), _Δ=_dx)
    curgrad_m1_stag_x[i] = curgrad_m1_stag_x[i] + adjvcur_x[i] * ∂p∂x

    return nothing
end