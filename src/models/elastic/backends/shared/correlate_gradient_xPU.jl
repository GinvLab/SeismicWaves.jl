@parallel function correlate_gradient_ρ_kernel!(grad_ρ, adjv, v_curr, v_old, _dt)
    @all(grad_ρ) = @all(grad_ρ) + (@all(adjv) * (@all(v_old) - @all(v_curr)) * _dt)

    return nothing
end

@parallel_indices (i, j) function correlate_gradient_λ_μ_kernel!(grad_λ, grad_μ, adjσxx, adjσzz, vx, vz, λ, μ, factx, factz, freetop)    
    ∂vx∂x = 0.0
    ∂vz∂z = 0.0
    if freetop && j == 1
        # on the free surface, σzz = 0
        ∂vx∂x = (-vx[i+1, j] + 27 * vx[i, j] - 27 * vx[i-1, j] + vx[i-2, j]) * factx
        # using boundary condition to calculate ∂vz∂z
        ∂vz∂z = -(λ[i, j] / (λ[i, j] + 2*μ[i, j])) * ∂vx∂x
    elseif freetop && j == 2
        # just below the free surface
        ∂vx∂x = (-vx[i+1, j] + 27 * vx[i, j] - 27 * vx[i-1, j] + vx[i-2, j]) * factx
        # zero velocity above the free surface
        ∂vz∂z = (-vz[i, j+1] + 27 * vz[i, j] - 27 * vz[i, j-1] + 0.0) * factz
    elseif j >= 3
        ∂vx∂x = (-vx[i+1, j] + 27 * vx[i, j] - 27 * vx[i-1, j] + vx[i-2, j]) * factx
        ∂vz∂z = (-vz[i, j+1] + 27 * vz[i, j] - 27 * vz[i, j-1] + vz[i, j-2]) * factz
    end
    # correlate
    grad_λ[i, j] += (-(∂vx∂x + ∂vz∂z) * (adjσxx[i, j] + adjσzz[i, j]))
    grad_μ[i, j] += (-2 * ∂vx∂x * adjσxx[i, j]) + (-2 * ∂vz∂z * adjσzz[i, j])

    return nothing
end

@parallel_indices (i, j) function correlate_gradient_μ_ihalf_jhalf_kernel!(grad_μ_ihalf_jhalf, adjσxz, vx, vz, factx, factz, freetop)
    ∂vx∂z = 0.0
    ∂vz∂x = 0.0
    if freetop && j == 1
        # zero velocity above the free surface
        ∂vx∂z = (-vx[i, j+2] + 27 * vx[i, j+1] - 27 * vx[i, j] + 0.0       ) * factz
        ∂vz∂x = (-vz[i+2, j] + 27 * vz[i+1, j] - 27 * vz[i, j] + vz[i-1, j]) * factx
    elseif j >= 2
        ∂vx∂z = (-vx[i, j+2] + 27 * vx[i, j+1] - 27 * vx[i, j] + vx[i, j-1]) * factz
        ∂vz∂x = (-vz[i+2, j] + 27 * vz[i+1, j] - 27 * vz[i, j] + vz[i-1, j]) * factx
    end
    # correlate
    grad_μ_ihalf_jhalf[i, j] += -(∂vx∂z+∂vz∂x) * adjσxz[i, j]

    return nothing
end

@views function correlate_gradients!(grid, vcurr, vold, dt, freetop)
    nx, nz = grid.size
    @parallel correlate_gradient_ρ_kernel!(
        grid.fields["grad_ρ_ihalf"].value,
        grid.fields["adjv"].value[1],
        vcurr[1],
        vold[1],
        1 / dt
    )
    @parallel correlate_gradient_ρ_kernel!(
        grid.fields["grad_ρ_jhalf"].value,
        grid.fields["adjv"].value[2],
        vcurr[2],
        vold[2],
        1 / dt
    )
    @parallel (3:nx-2, 1:nz-2) correlate_gradient_λ_μ_kernel!(
        grid.fields["grad_λ"].value,
        grid.fields["grad_μ"].value,
        grid.fields["adjσ"].value[1],
        grid.fields["adjσ"].value[2],
        vcurr...,
        grid.fields["λ"].value,
        grid.fields["μ"].value,
        (1.0 ./ (24.0 .* grid.spacing))...,
        freetop
    )
    @parallel (2:nx-2, 1:nz-2) correlate_gradient_μ_ihalf_jhalf_kernel!(
        grid.fields["grad_μ_ihalf_jhalf"].value,
        grid.fields["adjσ"].value[3],
        vcurr...,
        (1.0 ./ (24.0 .* grid.spacing))...,
        freetop
    )
end