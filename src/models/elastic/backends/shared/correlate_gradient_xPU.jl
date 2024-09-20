@parallel function correlate_gradient_ρ_kernel!(grad_ρ, adjv, v_curr, v_old, _dt)
    @all(grad_ρ) = @all(grad_ρ) + (@all(adjv) * (@all(v_old) - @all(v_curr)) * _dt)

    return nothing
end

@parallel_indices (i, j) function correlate_gradient_ihalf_kernel!(grad_λ_ihalf, grad_μ_ihalf, adjσxx, adjσzz, vx, vz, λ_ihalf, μ_ihalf, factx, factz, freetop)
    ∂vx∂x_fwd = ∂vz∂z_bkd = 0
    if freetop == true
        # j=1: we are on the free surface!
        if j == 1
            # vx derivative only in x so no problem
            ∂vx∂x_fwd = (vx[i-1, j] - 27.0 * vx[i, j] + 27.0 * vx[i+1, j] - vx[i+2, j]) * factx
            # using boundary condition to calculate ∂vz∂z_bkd from ∂vx∂x_fwd
            ∂vz∂z_bkd = -(λ_ihalf[i, j] / (λ_ihalf[i, j] + 2.0 * μ_ihalf[i, j])) * ∂vx∂x_fwd
        end
        # j=2: we are just below the surface (1/2)
        if j == 2
            # vx derivative only in x so no problem
            ∂vx∂x_fwd = (vx[i-1, j] - 27.0 * vx[i, j] + 27.0 * vx[i+1, j] - vx[i+2, j]) * factx
            # zero velocity above the free surface
            ∂vz∂z_bkd = (0.0 - 27.0 * vz[i, j-1] + 27.0 * vz[i, j] - vz[i, j+1]) * factz
        end
    end
    if j >= 3
        ∂vx∂x_fwd = (vx[i-1, j] - 27.0 * vx[i, j] + 27.0 * vx[i+1, j] - vx[i+2, j]) * factx
        ∂vz∂z_bkd = (vz[i, j-2] - 27.0 * vz[i, j-1] + 27.0 * vz[i, j] - vz[i, j+1]) * factz
    end
    # correlate
    grad_λ_ihalf[i, j] += -((∂vx∂x_fwd + ∂vz∂z_bkd) * (adjσxx[i, j] + adjσzz[i, j]))
    grad_μ_ihalf[i, j] += (-2 * ∂vx∂x_fwd * adjσxx[i, j]) + (-2 * ∂vz∂z_bkd * adjσzz[i, j])

    return nothing
end

@parallel_indices (i, j) function correlate_gradient_jhalf_kernel!(grad_μ_jhalf, adjσxz, vx, vz, factx, factz, freetop)
    ∂vx∂z_fwd = ∂vz∂x_bkd = 0
    if freetop
        if j == 1
            # zero velocity above the free surface
            ∂vx∂z_fwd = factz * (0.0 - 27.0 * vx[i, j] + 27.0 * vx[i, j+1] - vx[i, j+2])
            # vz derivative only in x so no problem
            ∂vz∂x_bkd = factx * (vz[i-2, j] - 27.0 * vz[i-1, j] + 27.0 * vz[i, j] - vz[i+1, j])
        end
    end
    if j >= 2
        ∂vx∂z_fwd = factz * (vx[i, j-1] - 27.0 * vx[i, j] + 27.0 * vx[i, j+1] - vx[i, j+2])
        ∂vz∂x_bkd = factx * (vz[i-2, j] - 27.0 * vz[i-1, j] + 27.0 * vz[i, j] - vz[i+1, j])
    end
    # correlate
    grad_μ_jhalf[i, j] += (-∂vx∂z_fwd-∂vz∂x_bkd) * adjσxz[i, j]

    return nothing
end

@views function correlate_gradients!(grid, vcurr, vold, dt, freetop)
    nx, nz = grid.size
    @parallel correlate_gradient_ρ_kernel!(grid.fields["grad_ρ"].value, grid.fields["adjv"].value[1], vcurr[1], vold[1], 1 / dt)
    @parallel correlate_gradient_ρ_kernel!(grid.fields["grad_ρ_ihalf_jhalf"].value, grid.fields["adjv"].value[2], vcurr[2], vold[2], 1 / dt)
    @parallel (2:nx-2, 1:nz-1) correlate_gradient_ihalf_kernel!(
        grid.fields["grad_λ_ihalf"].value,
        grid.fields["grad_μ_ihalf"].value,
        grid.fields["adjσ"].value[1], grid.fields["adjσ"].value[2], vcurr...,
        grid.fields["λ_ihalf"].value, grid.fields["μ_ihalf"].value,
        (1.0 ./ (24.0 .* grid.spacing))...,
        freetop
    )
    @parallel (3:nx-1, 1:nz-2) correlate_gradient_jhalf_kernel!(
        grid.fields["grad_μ_jhalf"].value,
        grid.fields["adjσ"].value[3], vcurr...,
        (1.0 ./ (24.0 .* grid.spacing))...,
        freetop
    )
end