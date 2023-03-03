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
    h_min = min(model.dx, model.dy)
    courant = vel_max * model.dt / h_min
    @debug "Courant number: $(courant)"
    @assert courant <= sqrt(2)/2 "Courant condition not satisfied!"
end

"""
    @views check_courant_condition(::IsotropicAcousticWaveEquation, model::WaveModel3D)

Check the Courant number for isotropic acoustic 3D models.
"""
@views function check_courant_condition(::IsotropicAcousticWaveEquation, model::WaveModel3D)
    vel_max = maximum(model.vel)
    h_min = min(model.dx, model.dy, model.dz)
    courant = vel_max * model.dt / h_min
    @debug "Courant number: $(courant)"
    @assert courant <= sqrt(3)/3 "Courant condition not satisfied!"
end

"""
    @views check_ppw(::IsotropicAcousticWaveEquation, model::WaveModel1D, srcs::Sources{<:Real}, min_ppw::Integer=10)

Check that the number of points per wavelength is above or equal to the specified threshold.
"""
@views function check_ppw(::IsotropicAcousticWaveEquation, model::WaveModel1D, srcs::Sources{<:Real}, min_ppw::Integer=10)
    vel_min = minimum(model.vel)
    ppw = vel_min / srcs.freqdomain / model.dx
    @debug "Points per wavelengh: $(ppw)"
    @assert ppw >= min_ppw "Not enough points per wavelengh!"
end

"""
    @views check_ppw(::IsotropicAcousticWaveEquation, model::WaveModel2D, srcs::Sources{<:Real}, ppw::Integer=10)

Check that the number of points per wavelength is above or equal to the specified threshold.
"""
@views function check_ppw(::IsotropicAcousticWaveEquation, model::WaveModel2D, srcs::Sources{<:Real}, min_ppw::Integer=10)
    vel_min = minimum(model.vel)
    h_max = max(model.dx, model.dy)
    ppw = vel_min / srcs.freqdomain / h_max
    @debug "Points per wavelengh: $(ppw)"
    @assert ppw >= min_ppw "Not enough points per wavelengh!"
end

"""
    @views check_ppw(::IsotropicAcousticWaveEquation, model::WaveModel3D, srcs::Sources{<:Real}, ppw::Integer=10)

Check that the number of points per wavelength is above or equal to the specified threshold.
"""
@views function check_ppw(::IsotropicAcousticWaveEquation, model::WaveModel3D, srcs::Sources{<:Real}, min_ppw::Integer=10)
    vel_min = minimum(model.vel)
    h_max = max(model.dx, model.dy, model.dz)
    ppw = vel_min / srcs.freqdomain / h_max
    @debug "Points per wavelengh: $(ppw)"
    @assert ppw >= min_ppw "Not enough points per wavelengh!"
end

@views function check_positions(::ReflectiveBoundaryCondition, model::WaveModel1D, positions::Matrix{<:Real})
    @assert size(positions, 2) == 1 "Positions matrix do not match the dimension of the model!"
    for s in 1:size(positions, 1)
        @assert 0 <= positions[s, 1] <= model.lx "Position $(s) is not inside the model!"
    end
end

@views function check_positions(::ReflectiveBoundaryCondition, model::WaveModel2D, positions::Matrix{<:Real})
    @assert size(positions, 2) == 2 "Positions matrix do not match the dimension of the model!"
    for s in 1:size(positions, 1)
        @assert (0 <= positions[s, 1] <= model.lx &&
                 0 <= positions[s, 2] <= model.ly ) "Position $(s) is not inside the model!"
    end
end

@views function check_positions(::ReflectiveBoundaryCondition, model::WaveModel3D, positions::Matrix{<:Real})
    @assert size(positions, 2) == 3 "Positions matrix do not match the dimension of the model!"
    for s in 1:size(positions, 1)
        @assert (0 <= positions[s, 1] <= model.lx &&
                 0 <= positions[s, 2] <= model.ly &&
                 0 <= positions[s, 3] <= model.lz ) "Position $(s) is not inside the model!"
    end
end

@views function check_positions(::CPMLBoundaryCondition, model::WaveModel1D, positions::Matrix{<:Real})
    check_positions(ReflectiveBoundaryCondition(), model, positions)
    for s in 1:size(positions, 1)
        @assert model.dx * model.halo <= positions[s, 1] <= model.lx - (model.dx * model.halo) "Position $(s) is inside the CPML region!"
    end
end

@views function check_positions(::CPMLBoundaryCondition, model::WaveModel2D, positions::Matrix{<:Real})
    check_positions(ReflectiveBoundaryCondition(), model, positions)
    for s in 1:size(positions, 1)
        @assert (model.dx * model.halo <= positions[s, 1] <= model.lx - (model.dx * model.halo) &&
                 model.dy * model.halo <= positions[s, 2] <= model.ly - (model.dy * model.halo)) "Position $(s) is inside the CPML region!"
    end
end

@views function check_positions(::CPMLBoundaryCondition, model::WaveModel3D, positions::Matrix{<:Real})
    check_positions(ReflectiveBoundaryCondition(), model, positions)
    for s in 1:size(positions, 1)
        @assert (model.dx * model.halo <= positions[s, 1] <= model.lx - (model.dx * model.halo) &&
                 model.dy * model.halo <= positions[s, 2] <= model.ly - (model.dy * model.halo) &&
                 model.dz * model.halo <= positions[s, 3] <= model.lz - (model.dz * model.halo)) "Position $(s) is inside the CPML region!"
    end
end

function check_invcov_matrix(model::WaveModel, invcov)
    @assert size(invcov) == (model.nt, model.nt) "Inverse of covariance matrix has not size equal to (nt x nt)!"
end

function check_checkpoint_frequency(model::WaveModel, check_freq)
    if check_freq !== nothing
        @assert check_freq > 2 "Checkpointing frequency must be bigger than 2!"
        @assert check_freq < model.nt "Checkpointing frequency must be smaller than the number of timesteps!"
    end
end
