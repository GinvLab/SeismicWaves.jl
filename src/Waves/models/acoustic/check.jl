"""
    @views check_courant_condition(::IsotropicAcousticWaveEquation, model::WaveModel2D)

Check the Courant number for isotropic acoustic 2D models.
"""
@views function check_courant_condition(::IsotropicAcousticWaveEquation, model::WaveModel2D)
    vel_max = maximum(model.vel)
    courant = vel_max * model.dt * sqrt(1 / model.dx^2 + 1 / model.dz^2)
    @info "Courant number: $(courant)"
    @assert courant <= 1.0 "Courant condition not satisfied!"
end

"""
    @views check_courant_condition(::IsotropicAcousticWaveEquation, model::WaveModel3D)

Check the Courant number for isotropic acoustic 3D models.
"""
@views function check_courant_condition(::IsotropicAcousticWaveEquation, model::WaveModel3D)
    vel_max = maximum(model.vel)
    courant = vel_max * model.dt * sqrt(1 / model.dx^2 + 1 / model.dy^2 + 1 / model.dz^2)
    @info "Courant number: $(courant)"
    @assert courant <= 1.0 "Courant condition not satisfied!"
end

"""
    @views check_ppw(::IsotropicAcousticWaveEquation, model::WaveModel2D, srcs::Sources{<:Real}, ppw::Integer=10)

Check that the number of points per wavelength is above or equal to the specified threshold.
"""
@views function check_ppw(::IsotropicAcousticWaveEquation, model::WaveModel2D, srcs::Sources{<:Real}, ppw::Integer=10)
    vel_max = maximum(model.vel)
    @assert model.dx <= vel_max / (ppw * srcs.freqdomain) "Not enough points per wavelengh in x-direction!"
    @assert model.dz <= vel_max / (ppw * srcs.freqdomain) "Not enough points per wavelengh in z-direction!"
end

"""
    @views check_ppw(::IsotropicAcousticWaveEquation, model::WaveModel3D, srcs::Sources{<:Real}, ppw::Integer=10)

Check that the number of points per wavelength is above or equal to the specified threshold.
"""
@views function check_ppw(::IsotropicAcousticWaveEquation, model::WaveModel3D, srcs::Sources{<:Real}, ppw::Integer=10)
    vel_max = maximum(model.vel)
    @assert model.dx <= vel_max / (ppw * srcs.freqdomain) "Not enough points per wavelengh in x-direction!"
    @assert model.dy <= vel_max / (ppw * srcs.freqdomain) "Not enough points per wavelengh in y-direction!"
    @assert model.dz <= vel_max / (ppw * srcs.freqdomain) "Not enough points per wavelengh in z-direction!"
end
