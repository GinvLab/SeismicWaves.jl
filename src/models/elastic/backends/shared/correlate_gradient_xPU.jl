@parallel function correlate_gradient_ρ_kernel!(curgrad_ρ, adjv, v_curr, v_old, v_veryold, _dt2)
    @all(curgrad_ρ) = @all(curgrad_ρ) + (@all(adjv) * (@all(v_curr) - 2 * @all(v_old) + @all(v_veryold)) * _dt2)

    return nothing
end

@views function correlate_gradient_ρ!(curgrad_ρ, adjv, v_curr, v_old, v_veryold, dt)
    _dt2 = 1 / dt^2
    adjvx = adjv[1]
    vx_curr = v_curr[1]
    vx_old = v_old[1]
    vx_veryold = v_veryold[1]
    @parallel correlate_gradient_ρ_kernel!(curgrad_ρ, adjvx, vx_curr, vx_old, vx_veryold, _dt2)
end

@views function correlate_gradient_ρ_ihalf_jhalf!(curgrad_ρ_ihalf_jhalf, adjv, v_curr, v_old, v_veryold, dt)
    _dt2 = 1 / dt^2
    adjvz = adjv[2]
    vz_curr = v_curr[2]
    vz_old = v_old[2]
    vz_veryold = v_veryold[2]
    @parallel correlate_gradient_ρ_kernel!(curgrad_ρ_ihalf_jhalf, adjvz, vz_curr, vz_old, vz_veryold, _dt2)
end
