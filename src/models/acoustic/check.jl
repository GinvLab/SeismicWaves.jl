@views function check_courant_condition(::AcousticWaveEquation, model::WaveModel1D)
    vel_max = get_maximum_func(model)(model.vel)
    courant = vel_max * model.dt / model.Δs[1]
    @debug "Courant number: $(courant)"
    @assert courant <= 1.0 "Courant condition not satisfied!"
end

@views function check_courant_condition(::AcousticWaveEquation, model::WaveModel2D)
    vel_max = get_maximum_func(model)(model.vel)
    h_min = min(model.Δs[1], model.Δs[2])
    courant = vel_max * model.dt / h_min
    @debug "Courant number: $(courant)"
    @assert courant <= sqrt(2)/2 "Courant condition not satisfied!"
end

@views function check_courant_condition(::AcousticWaveEquation, model::WaveModel3D)
    vel_max = get_maximum_func(model)(model.vel)
    h_min = min(model.Δs[1], model.Δs[2], model.Δs[3])
    courant = vel_max * model.dt / h_min
    @debug "Courant number: $(courant)"
    @assert courant <= sqrt(3)/3 "Courant condition not satisfied!"
end

@views function check_ppw(::AcousticWaveEquation, model::WaveModel1D, srcs::Sources{<:Real}, min_ppw::Integer=10)
    vel_min = get_minimum_func(model)(model.vel)
    ppw = vel_min / srcs.freqdomain / model.Δs[1]
    @debug "Points per wavelengh: $(ppw)"
    @assert ppw >= min_ppw "Not enough points per wavelengh!"
end

@views function check_ppw(::AcousticWaveEquation, model::WaveModel2D, srcs::Sources{<:Real}, min_ppw::Integer=10)
    vel_min = get_minimum_func(model)(model.vel)
    h_max = max(model.Δs[1], model.Δs[2])
    ppw = vel_min / srcs.freqdomain / h_max
    @debug "Points per wavelengh: $(ppw)"
    @assert ppw >= min_ppw "Not enough points per wavelengh!"
end

@views function check_ppw(::AcousticWaveEquation, model::WaveModel3D, srcs::Sources{<:Real}, min_ppw::Integer=10)
    vel_min = get_minimum_func(model)(model.vel)
    h_max = max(model.Δs[1], model.Δs[2], model.Δs[3])
    ppw = vel_min / srcs.freqdomain / h_max
    @debug "Points per wavelengh: $(ppw)"
    @assert ppw >= min_ppw "Not enough points per wavelengh!"
end

@views function check_positions(::ReflectiveBoundaryCondition, model::WaveModel1D, positions::Matrix{<:Real})
    @assert size(positions, 2) == 1 "Positions matrix do not match the dimension of the model!"
    for s in 1:size(positions, 1)
        @assert 0 <= positions[s, 1] <= model.ls[1] "Position $(s) is not inside the model!"
    end
end

@views function check_positions(::ReflectiveBoundaryCondition, model::WaveModel2D, positions::Matrix{<:Real})
    @assert size(positions, 2) == 2 "Positions matrix do not match the dimension of the model!"
    for s in 1:size(positions, 1)
        @assert (0 <= positions[s, 1] <= model.ls[1] &&
                 0 <= positions[s, 2] <= model.ls[2] ) "Position $(s) is not inside the model!"
    end
end

@views function check_positions(::ReflectiveBoundaryCondition, model::WaveModel3D, positions::Matrix{<:Real})
    @assert size(positions, 2) == 3 "Positions matrix do not match the dimension of the model!"
    for s in 1:size(positions, 1)
        @assert (0 <= positions[s, 1] <= model.ls[1] &&
                 0 <= positions[s, 2] <= model.ls[2] &&
                 0 <= positions[s, 3] <= model.ls[3] ) "Position $(s) is not inside the model!"
    end
end

@views function check_positions(::CPMLBoundaryCondition, model::WaveModel1D, positions::Matrix{<:Real})
    check_positions(ReflectiveBoundaryCondition(), model, positions)
    for s in 1:size(positions, 1)
        @assert model.Δs[1] * model.halo <= positions[s, 1] <= model.ls[1] - (model.Δs[1] * model.halo) "Position $(s) is inside the CPML region!"
    end
end

@views function check_positions(::CPMLBoundaryCondition, model::WaveModel2D, positions::Matrix{<:Real})
    check_positions(ReflectiveBoundaryCondition(), model, positions)
    for s in 1:size(positions, 1)
        @assert (model.Δs[1] * model.halo <= positions[s, 1] <= model.ls[1] - (model.Δs[1] * model.halo) &&
                 (model.freetop ? 0 : model.Δs[2] * model.halo) <= positions[s, 2] <= model.ls[2] - (model.Δs[2] * model.halo)) "Position $(s) is inside the CPML region!"
    end
end

@views function check_positions(::CPMLBoundaryCondition, model::WaveModel3D, positions::Matrix{<:Real})
    check_positions(ReflectiveBoundaryCondition(), model, positions)
    for s in 1:size(positions, 1)
        @assert (model.Δs[1] * model.halo <= positions[s, 1] <= model.ls[1] - (model.Δs[1] * model.halo) &&
                 model.Δs[2] * model.halo <= positions[s, 2] <= model.ls[2] - (model.Δs[2] * model.halo) &&
                 (model.freetop ? 0 : model.Δs[3] * model.halo) <= positions[s, 3] <= model.ls[3] - (model.Δs[3] * model.halo)) "Position $(s) is inside the CPML region!"
    end
end

function check_invcov_matrix(model::WaveModel, invcov)
    @assert size(invcov) == (model.nt, model.nt) "Inverse of covariance matrix has not size equal to ($(model.nt) x $(model.nt))!"
end

function check_checkpoint_frequency(model::WaveModel, check_freq)
    if check_freq !== nothing
        @assert check_freq > 2 "Checkpointing frequency must be bigger than 2!"
        @assert check_freq < model.nt "Checkpointing frequency must be smaller than the number of timesteps!"
    end
end
