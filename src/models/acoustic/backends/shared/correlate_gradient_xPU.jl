@parallel function correlate_gradient_kernel!(curgrad, adjcur, pcur, pold, pveryold, _dt2)
    @all(curgrad) = @all(curgrad) + (@all(adjcur) * (@all(pcur) - 2.0 * @all(pold) + @all(pveryold)) * _dt2)

    return nothing
end

@views function correlate_gradient!(curgrad, adjcur, pcur, pold, pveryold, dt)
    _dt2 = 1 / dt^2
    @parallel correlate_gradient_kernel!(curgrad, adjcur, pcur, pold, pveryold, _dt2)
end
