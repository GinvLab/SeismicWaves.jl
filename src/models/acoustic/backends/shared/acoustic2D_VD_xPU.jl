@parallel_indices (is) function inject_sources!(pcur, srctf, possrcs, it)
    isrc = floor(Int, possrcs[is, 1])
    jsrc = floor(Int, possrcs[is, 2])
    pcur[isrc, jsrc] += srctf[it, is]

    return nothing
end

@parallel_indices (ir) function record_receivers!(pcur, traces, posrecs, it)
    irec = floor(Int, posrecs[ir, 1])
    jrec = floor(Int, posrecs[ir, 2])
    traces[it, ir] = pcur[irec, jrec]

    return nothing
end

@parallel_indices (i, j) function update_p_CPML!(
    pcur, vx_cur, vy_cur, fact_m0, _dx, _dy,
    halo, ξ_x, ξ_y, b_x, b_y, a_x, a_y
)
    ##################################
    # ∂p/∂t = (∂vx/∂x + ∂vy/∂y) / m0 #
    # fact_m0 = 1 / m0 / dt          #
    ##################################

    # Compute velocity derivatives
    ∂vx∂x = @∂̃x(vx_cur, a_x, b_x, ξ_x,
                order=4, I=(i-1,j), _Δ=_dx,
                halo=halo, halfgrid=false)
    ∂vy∂y = @∂̃y(vy_cur, a_y, b_y, ξ_y,
                order=4, I=(i,j-1), _Δ=_dy,
                halo=halo, halfgrid=false)
    # Update pressure
    pcur[i, j] -= fact_m0[i, j] * (∂vx∂x + ∂vy∂y)

    return nothing
end

@parallel_indices (i, j) function update_vx_CPML!(
    pcur, vx_cur, fact_m1_x, _dx,
    halo, ψ_x, b_x_half, a_x_half
)
    ########################
    # ∂vx/∂t = -∂p/∂x * m1 #
    # fact_m1_x = m1 / dt  #
    ########################

    # Compute pressure derivative in x direction
    ∂p∂x = @∂̃x(pcur, a_x_half, b_x_half, ψ_x,
               order=4, I=(i,j), _Δ=_dx,
               halo=halo, halfgrid=true)
    # Update velocity
    vx_cur[i, j] -= fact_m1_x[i, j] * ∂p∂x

    return nothing
end

@parallel_indices (i, j) function update_vy_CPML!(
    pcur, vy_cur, fact_m1_y, _dy,
    halo, ψ_y, b_y_half, a_y_half
)
    ########################
    # ∂vy/∂t = -∂p/∂y * m1 #
    # fact_m1_y = m1 / dt  #
    ########################

    # Compute pressure derivative in y direction
    ∂p∂y = @∂̃y(pcur, a_y_half, b_y_half, ψ_y,
               order=4, I=(i,j), _Δ=_dy,
               halo=halo, halfgrid=true)
    # Update velocity
    vy_cur[i, j] -= fact_m1_y[i, j] * ∂p∂y

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
    model, possrcs, srctf, posrecs, traces, it;
    save_trace=true
)
    # Extract info from grid
    grid = model.grid
    nx, ny = grid.size
    dx, dy = grid.spacing
    pcur = grid.fields["pcur"].value
    vx_cur, vy_cur = grid.fields["vcur"].value
    fact_m0 = grid.fields["fact_m0"].value
    fact_m1_x, fact_m1_y = grid.fields["fact_m1_stag"].value
    ψ_x, ψ_y = grid.fields["ψ"].value
    ξ_x, ξ_y = grid.fields["ξ"].value
    a_x = model.cpmlcoeffs[1].a
    a_x_half = model.cpmlcoeffs[1].a_h
    b_x = model.cpmlcoeffs[1].b
    b_x_half = model.cpmlcoeffs[1].b_h
    a_y = model.cpmlcoeffs[2].a
    a_y_half = model.cpmlcoeffs[2].a_h
    b_y = model.cpmlcoeffs[2].b
    b_y_half = model.cpmlcoeffs[2].b_h
    halo = model.cpmlparams.halo
    # Precompute divisions
    _dx = 1 / dx
    _dy = 1 / dy

    @parallel (2:(nx-1),2:(ny-1)) update_p_CPML!(
        pcur, vx_cur, vy_cur, fact_m0, _dx, _dy,
        halo, ξ_x, ξ_y, b_x, b_y, a_x, a_y
    )
    @parallel (1:size(possrcs, 1)) inject_sources!(pcur, srctf, possrcs, it)

    @parallel_async (1:(nx-1), 1:ny) update_vx_CPML!(
        pcur, vx_cur, fact_m1_x, _dx,
        halo, ψ_x, b_x_half, a_x_half
    )
    @parallel_async (1:nx, 1:(ny-1)) update_vy_CPML!(
        pcur, vy_cur, fact_m1_y, _dy,
        halo, ψ_y, b_y_half, a_y_half
    )
    @synchronize

    if save_trace
        @parallel (1:size(posrecs, 1)) record_receivers!(pcur, traces, posrecs, it)
    end
end

function adjoint_onestep_CPML!(model, possrcs, srctf, it)
    # Extract info from grid
    grid = model.grid
    nx, ny = grid.size
    dx, dy = grid.spacing
    pcur = grid.fields["adjpcur"].value
    vx_cur, vy_cur = grid.fields["adjvcur"].value
    fact_m0 = grid.fields["fact_m0"].value
    fact_m1_x, fact_m1_y = grid.fields["fact_m1_stag"].value
    ψ_x, ψ_y = grid.fields["ψ_adj"].value
    ξ_x, ξ_y = grid.fields["ξ_adj"].value
    a_x = model.cpmlcoeffs[1].a
    a_x_half = model.cpmlcoeffs[1].a_h
    b_x = model.cpmlcoeffs[1].b
    b_x_half = model.cpmlcoeffs[1].b_h
    a_y = model.cpmlcoeffs[2].a
    a_y_half = model.cpmlcoeffs[2].a_h
    b_y = model.cpmlcoeffs[2].b
    b_y_half = model.cpmlcoeffs[2].b_h
    halo = model.cpmlparams.halo
    # Precompute divisions
    _dx = 1 / dx
    _dy = 1 / dy

    @parallel_async (1:(nx-1), 1:ny) update_vx_CPML!(
        pcur, vx_cur, fact_m1_x, _dx,
        halo, ψ_x, b_x_half, a_x_half
    )
    @parallel_async (1:nx, 1:(ny-1)) update_vy_CPML!(
        pcur, vy_cur, fact_m1_y, _dy,
        halo, ψ_y, b_y_half, a_y_half
    )
    @synchronize

    @parallel (2:(nx-1),2:(ny-1)) update_p_CPML!(
        pcur, vx_cur, vy_cur, fact_m0, _dx, _dy,
        halo, ξ_x, ξ_y, b_x, b_y, a_x, a_y
    )
    @parallel (1:size(possrcs, 1)) inject_sources!(pcur, srctf, possrcs, it)
end

@views function correlate_gradient_m1!(curgrad_m1_stag, adjvcur, pold, gridspacing)
    _gridspacing = 1 ./ gridspacing
    nx, ny = size(pold)
    @parallel (1:(nx-1), 1:ny) correlate_gradient_m1_kernel_x!(curgrad_m1_stag[1], adjvcur[1], pold, _gridspacing[1])
    @parallel (1:nx, 1:(ny-1)) correlate_gradient_m1_kernel_y!(curgrad_m1_stag[2], adjvcur[2], pold, _gridspacing[2])
end

@parallel_indices (i, j) function correlate_gradient_m1_kernel_x!(curgrad_m1_stag_x, adjvcur_x, pold, _dx)
    ∂p∂x = @∂x(pold, order=4, I=(i,j), _Δ=_dx)
    curgrad_m1_stag_x[i, j] = curgrad_m1_stag_x[i, j] + adjvcur_x[i, j] * ∂p∂x

    return nothing
end

@parallel_indices (i, j) function correlate_gradient_m1_kernel_y!(curgrad_m1_stag_y, adjvcur_y, pold, _dy)
    ∂p∂y = @∂y(pold, order=4, I=(i,j), _Δ=_dy)
    curgrad_m1_stag_y[i, j] = curgrad_m1_stag_y[i, j] + adjvcur_y[i, j] * ∂p∂y

    return nothing
end