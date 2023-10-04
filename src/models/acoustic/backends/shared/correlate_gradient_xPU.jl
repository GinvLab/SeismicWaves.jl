@parallel function correlate_gradient_kernel!(curgrad, adjcur, pcur, pold, pveryold, _dt2)
    @all(curgrad) = @all(curgrad) + (@all(adjcur) * (@all(pcur) - 2.0 * @all(pold) + @all(pveryold)) * _dt2)

    return nothing
end

@views function correlate_gradient!(curgrad, adjcur, pcur, pold, pveryold, dt)
    _dt2 = 1 / dt^2
    @parallel correlate_gradient_kernel!(curgrad, adjcur, pcur, pold, pveryold, _dt2)
end

@parallel function correlate_gradient_m0_kernel!(curgrad_m0, adjpcur, pcur_corr, pcur_old, _dt)
    @all(curgrad_m0) = @all(curgrad_m0) + (@all(adjpcur) * (@all(pcur_corr) - @all(pcur_old)) * _dt)

    return nothing
end

@views function correlate_gradient_m0!(curgrad_m0, adjpcur, pcur_corr, pcur_old, dt)
    _dt = 1 / dt
    @parallel correlate_gradient_m0_kernel!(curgrad_m0, adjpcur, pcur_corr, pcur_old, _dt)
end

# @parallel function correlate_gradient_m1_kernel!(curgrad_m1_stag_x, curgrad_m1_stag_y, curgrad_m1_stag_z,
#                                                  adjvcur_x, adjvcur_y, adjvcur_z, pold, _dx, _dy, _dz)
#     @all(curgrad_m1_stag_x) = @all(curgrad_m1_stag_x) + (@all(adjvcur_x) * @d_dxa(pold) * _dx)
#     @all(curgrad_m1_stag_y) = @all(curgrad_m1_stag_y) + (@all(adjvcur_y) * @d_dya(pold) * _dy)
#     @all(curgrad_m1_stag_z) = @all(curgrad_m1_stag_z) + (@all(adjvcur_z) * @d_dza(pold) * _dz)
#     return nothing
# end

@views function correlate_gradient_m1!(curgrad_m1_stag, adjvcur, pold, gridspacing)
    _gridspacing = 1 ./ gridspacing
    @parallel correlate_gradient_m1_kernel!(curgrad_m1_stag..., adjvcur..., pold, _gridspacing...)
end
