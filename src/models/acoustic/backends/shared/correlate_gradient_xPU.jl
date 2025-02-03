@parallel function correlate_gradient_kernel!(curgrad, adjcur, pcur, pold, pveryold, _dt2)
    @all(curgrad) = @all(curgrad) + (@all(adjcur) * (@all(pcur) - 2.0 * @all(pold) + @all(pveryold)) * _dt2)

    return nothing
end

function correlate_gradient!(curgrad, adjcur, pcur, pold, pveryold, dt)
    _dt2 = 1 / dt^2
    @parallel correlate_gradient_kernel!(curgrad, adjcur, pcur, pold, pveryold, _dt2)
end

@parallel function correlate_gradient_m0_kernel!(curgrad_m0, adjpcur, pcur_corr, pcur_old, _dt)
    @all(curgrad_m0) = @all(curgrad_m0) - (@all(adjpcur) * (@all(pcur_corr) - @all(pcur_old)) * _dt)

    return nothing
end

function correlate_gradient_m0!(curgrad_m0, adjpcur, pcur_corr, pcur_old, dt)
    _dt = 1 / dt
    @parallel correlate_gradient_m0_kernel!(curgrad_m0, adjpcur, pcur_corr, pcur_old, _dt)
end
