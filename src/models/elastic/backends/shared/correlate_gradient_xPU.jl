@parallel function correlate_gradient_ρ_kernel!(grad_ρ, adjcur, uold, ucur, unew, _dt2)
    @all(grad_ρ) = @all(grad_ρ) + (@all(adjcur) * (@all(uold) - 2 * @all(ucur) + @all(unew)) * _dt2)

    return nothing
end

@parallel_indices (i, j) function correlate_gradient_λ_μ_kernel!(grad_λ, grad_μ, adjux, adjuz, ux, uz, λ, μ, _Δx, _Δz, nx, nz)
    # Compute bulk strain tensor components
    ε_xx     = ∂ux∂x_4th(ux   , uz   , λ, μ, i, j, _Δx, _Δz, nx, nz)
    ε_xx_adj = ∂ux∂x_4th(adjux, adjuz, λ, μ, i, j, _Δx, _Δz, nx, nz)
    ε_zz     = ∂uz∂z_4th(ux   , uz   , λ, μ, i, j, _Δx, _Δz, nx, nz)
    ε_zz_adj = ∂uz∂z_4th(adjux, adjuz, λ, μ, i, j, _Δx, _Δz, nx, nz)
    # Compute divergence of displacement
    div_u = ε_xx + ε_zz
    div_u_adj = ε_xx_adj + ε_zz_adj
    # Accumulate gradients
    grad_λ[i, j] += div_u * div_u_adj
    grad_μ[i, j] += 2 * (ε_xx * ε_xx_adj + ε_zz * ε_zz_adj)

    return nothing
end

@parallel_indices (i, j) function correlate_gradient_μ_ihalf_jhalf_kernel!(grad_μ_ihalf_jhalf, adjux, adjuz, ux, uz, _Δx, _Δz, nx, nz)
    # Compute shear strain tensor components
    ε_xz     = (∂uz∂x_4th(uz   , i, j, _Δx, nx) + 
                ∂ux∂z_4th(ux   , i, j, _Δz, nz)) / 2
    ε_xz_adj = (∂uz∂x_4th(adjuz, i, j, _Δx, nx) + 
                ∂ux∂z_4th(adjux, i, j, _Δz, nz)) / 2
    ε_zx = ε_xz
    ε_zx_adj = ε_xz_adj
    # Accumulate gradients
    grad_μ_ihalf_jhalf[i, j] += 2 * (ε_xz * ε_xz_adj + ε_zx * ε_zx_adj)

    return nothing
end

function correlate_gradients!(grid, uold_corr, ucur_corr, unew_corr, dt)
    nx, nz = grid.size
    @parallel correlate_gradient_ρ_kernel!(
        grid.fields["grad_ρ_ihalf"].value,
        grid.fields["adjucur"].value[1],
        uold_corr[1], ucur_corr[1], unew_corr[1],
        1 / dt^2
    )
    @parallel correlate_gradient_ρ_kernel!(
        grid.fields["grad_ρ_jhalf"].value,
        grid.fields["adjucur"].value[2],
        uold_corr[2], ucur_corr[2], unew_corr[2],
        1 / dt^2
    )
    @parallel (1:nx, 1:nz) correlate_gradient_λ_μ_kernel!(
        grid.fields["grad_λ"].value,
        grid.fields["grad_μ"].value,
        grid.fields["adjucur"].value...,
        ucur_corr...,
        grid.fields["λ"].value,
        grid.fields["μ"].value,
        (1 ./ grid.spacing)...,
        grid.size...
    )
    @parallel (1:nx-1, 1:nz-1) correlate_gradient_μ_ihalf_jhalf_kernel!(
        grid.fields["grad_μ_ihalf_jhalf"].value,
        grid.fields["adjucur"].value...,
        ucur_corr...,
        (1 ./ grid.spacing)...,
        grid.size...
    )
end