"""
    @views check_courant_condition(::IsotropicAcousticWaveEquation, model::WaveModel1D)

Check the Courant number for isotropic acoustic 1D models.
"""
@views function check_courant_condition(::IsotropicAcousticWaveEquation, model::WaveModel1D)
    vel_max = maximum(model.vel)
    courant = vel_max * model.dt / model.dx
    @debug "Courant number: $(courant)"
    @assert courant <= 1.0 "Courant condition not satisfied!"
end

"""
    @views check_courant_condition(::IsotropicAcousticWaveEquation, model::WaveModel2D)

Check the Courant number for isotropic acoustic 2D models.
"""
@views function check_courant_condition(::IsotropicAcousticWaveEquation, model::WaveModel2D)
    vel_max = maximum(model.vel)
    courant = vel_max * model.dt * (1/model.dx + 1/model.dy)
    @debug "Courant number: $(courant)"
    @assert courant <= sqrt(2)/2 "Courant condition not satisfied!"
end

"""
    @views check_courant_condition(::IsotropicAcousticWaveEquation, model::WaveModel3D)

Check the Courant number for isotropic acoustic 3D models.
"""
@views function check_courant_condition(::IsotropicAcousticWaveEquation, model::WaveModel3D)
    vel_max = maximum(model.vel)
    courant = vel_max * model.dt * (1/model.dx + 1/model.dy + 1/model.dz)
    @debug "Courant number: $(courant)"
    @assert courant <= sqrt(3)/3 "Courant condition not satisfied!"
end

"""
    @views check_ppw(::IsotropicAcousticWaveEquation, model::WaveModel1D, srcs::Sources{<:Real}, min_ppw::Integer=10)

Check that the number of points per wavelength is above or equal to the specified threshold.
"""
@views function check_ppw(::IsotropicAcousticWaveEquation, model::WaveModel1D, srcs::Sources{<:Real}, min_ppw::Integer=10)
    vel_min = minimum(model.vel)
    max_cell_size = model.dx
    ppw = vel_min / srcs.freqdomain / max_cell_size
    @debug "Points per wavelengh: $(ppw)"
    @assert ppw >= min_ppw "Not enough points per wavelengh!"
end

"""
    @views check_ppw(::IsotropicAcousticWaveEquation, model::WaveModel2D, srcs::Sources{<:Real}, ppw::Integer=10)

Check that the number of points per wavelength is above or equal to the specified threshold.
"""
@views function check_ppw(::IsotropicAcousticWaveEquation, model::WaveModel2D, srcs::Sources{<:Real}, min_ppw::Integer=10)
    vel_min = minimum(model.vel)
    max_cell_size = sqrt(model.dx^2 + model.dy^2)
    ppw = vel_min / srcs.freqdomain / max_cell_size
    @debug "Points per wavelengh: $(ppw)"
    @assert ppw >= min_ppw "Not enough points per wavelengh!"
end

"""
    @views check_ppw(::IsotropicAcousticWaveEquation, model::WaveModel3D, srcs::Sources{<:Real}, ppw::Integer=10)

Check that the number of points per wavelength is above or equal to the specified threshold.
"""
@views function check_ppw(::IsotropicAcousticWaveEquation, model::WaveModel3D, srcs::Sources{<:Real}, min_ppw::Integer=10)
    vel_min = minimum(model.vel)
    max_cell_size = sqrt(model.dx^2 + model.dy^2 + model.dz^2)
    ppw = vel_min / srcs.freqdomain / max_cell_size
    @debug "Points per wavelengh: $(ppw)"
    @assert ppw >= min_ppw "Not enough points per wavelengh!"
end

@views function check_positions(model::WaveModel1D, positions::Matrix{<:Real})
    @assert size(positions, 2) == 1 "Positions matrix do not match the dimension of the model!"
    for s in 1:size(positions, 1)
        @assert 0 <= positions[s, 1] <= model.lx "Position $(s) is not inside the model!"
    end
end