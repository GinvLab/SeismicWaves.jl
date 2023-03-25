

#######################################

@views function check_courant_condition(model::Acoustic_CD_WaveSimul)
    vel_max = get_maximum_func(model)(model.vel)
    tmp = sqrt(sum(1 ./ model.gridspacing.^2))
    courant = vel_max * model.dt / tmp    
    @debug "Courant number: $(courant)"
    @assert courant <= 1.0 "Courant condition not satisfied!"
end

#####################################

# @views function check_courant_condition(model::Acoustic_CD_WaveSimul{1})
#     vel_max = get_maximum_func(model)(model.vel)
#     courant = vel_max * model.dt / model.gridspacing[1]
#     @debug "Courant number: $(courant)"
#     @assert courant <= 1.0 "Courant condition not satisfied!"
# end

# @views function check_courant_condition(model::Acoustic_CD_WaveSimul{2})
#     vel_max = get_maximum_func(model)(model.vel)
#     h_min = min(model.gridspacing[1], model.gridspacing[2])
#     courant = vel_max * model.dt / h_min
#     @debug "Courant number: $(courant)"
#     @assert courant <= sqrt(2)/2 "Courant condition not satisfied!"
# end

# @views function check_courant_condition(model::Acoustic_CD_WaveSimul{3})
#     vel_max = get_maximum_func(model)(model.vel)
#     h_min = min(model.gridspacing[1], model.gridspacing[2], model.gridspacing[3])
#     courant = vel_max * model.dt / h_min
#     @debug "Courant number: $(courant)"
#     @assert courant <= sqrt(3)/3 "Courant condition not satisfied!"
# end

@views function check_ppw(model::Acoustic_CD_WaveSimul, srcs::Sources{<:Real}, min_ppw::Integer=10)
    vel_min = get_minimum_func(model)(model.vel)
    h_max = maximum(model.gridspacing)
    ppw = vel_min / srcs.domfreq / h_max
    @debug "Points per wavelengh: $(ppw)"
    @assert ppw >= min_ppw "Not enough points per wavelengh!"
end

# @views function check_ppw(model::Acoustic_CD_WaveSimul{1}, srcs::Sources{<:Real}, min_ppw::Integer=10)
#     vel_min = get_minimum_func(model)(model.vel)
#     ppw = vel_min / srcs.domfreq / model.gridspacing[1]
#     @debug "Points per wavelengh: $(ppw)"
#     @assert ppw >= min_ppw "Not enough points per wavelengh!"
# end

# @views function check_ppw(model::Acoustic_CD_WaveSimul{2}, srcs::Sources{<:Real}, min_ppw::Integer=10)
#     vel_min = get_minimum_func(model)(model.vel)
#     h_max = max(model.gridspacing[1], model.gridspacing[2])
#     ppw = vel_min / srcs.domfreq / h_max
#     @debug "Points per wavelengh: $(ppw)"
#     @assert ppw >= min_ppw "Not enough points per wavelengh!"
# end

# @views function check_ppw(model::Acoustic_CD_WaveSimul{3}, model::WaveSimul3D, srcs::Sources{<:Real}, min_ppw::Integer=10)
#     vel_min = get_minimum_func(model)(model.vel)
#     h_max = max(model.gridspacing[1], model.gridspacing[2], model.gridspacing[3])
#     ppw = vel_min / srcs.domfreq / h_max
#     @debug "Points per wavelengh: $(ppw)"
#     @assert ppw >= min_ppw "Not enough points per wavelengh!"
# end

####################################################

function check_positions(model::Acoustic_CD_CPML_WaveSimul, positions::Matrix{<:Real})
    ndimwavsim = length(model.gridspacing)
    @assert size(positions, 2) == ndimwavsim "Positions matrix do not match the dimension of the model!"

    for s in 1:size(positions, 1)
        for c=1:size(positions, 2)
            @assert (0 <= positions[s, c] <= model.ls[c]) "Position $(s) is not inside the grid!"
            @assert (model.gridspacing[c] * model.halo <= positions[s, c] <= model.ls[c] - (model.gridspacing[c] * model.halo) ) "Position $(s) is inside the CPML region!"
        end
    end
    return
end

####################################################


function check_invcov_matrix(model::WaveSimul, invcov)
    @assert size(invcov) == (model.nt, model.nt) "Inverse of covariance matrix has not size equal to ($(model.nt) x $(model.nt))!"
end

function check_checkpoint_frequency(model::WaveSimul, check_freq)
    if check_freq !== nothing
        @assert check_freq > 2 "Checkpointing frequency must be bigger than 2!"
        @assert check_freq < model.nt "Checkpointing frequency must be smaller than the number of timesteps!"
    end
end

#####################################################

# @views function check_positions(model::Acoustic_CD_WaveSimul{1}, positions::Matrix{<:Real})
#     @assert size(positions, 2) == 1 "Positions matrix do not match the dimension of the model!"
#     for s in 1:size(positions, 1)
#         @assert 0 <= positions[s, 1] <= model.ls[1] "Position $(s) is not inside the model!"
#     end
# end

# @views function check_positions(model::Acoustic_CD__WaveSimul{2}, positions::Matrix{<:Real})
#     @assert size(positions, 2) == 2 "Positions matrix do not match the dimension of the model!"
#     for s in 1:size(positions, 1)
#         @assert (0 <= positions[s, 1] <= model.ls[1] &&
#                  0 <= positions[s, 2] <= model.ls[2] ) "Position $(s) is not inside the model!"
#     end
# end

# @views function check_positions(model::Acoustic_CD_Refl_WaveSimul{3}, positions::Matrix{<:Real})
#     @assert size(positions, 2) == 3 "Positions matrix do not match the dimension of the model!"
#     for s in 1:size(positions, 1)
#         @assert (0 <= positions[s, 1] <= model.ls[1] &&
#                  0 <= positions[s, 2] <= model.ls[2] &&
#                  0 <= positions[s, 3] <= model.ls[3] ) "Position $(s) is not inside the model!"
#     end
# end

# @views function check_positions(model::Acoustic_CD_CPML_WaveSimul{1}, positions::Matrix{<:Real})
#     check_positions(ReflectiveBoundaryCondition(), model, positions)
#     for s in 1:size(positions, 1)
#         @assert model.gridspacing[1] * model.halo <= positions[s, 1] <= model.ls[1] - (model.gridspacing[1] * model.halo) "Position $(s) is inside the CPML region!"
#     end
# end

# @views function check_positions(model::Acoustic_CD_CPML_WaveSimul{2}, positions::Matrix{<:Real})
#     check_positions(ReflectiveBoundaryCondition(), model, positions)
#     for s in 1:size(positions, 1)
#         @assert (model.gridspacing[1] * model.halo <= positions[s, 1] <= model.ls[1] - (model.gridspacing[1] * model.halo) &&
#                  (model.freetop ? 0 : model.gridspacing[2] * model.halo) <= positions[s, 2] <= model.ls[2] - (model.gridspacing[2] * model.halo)) "Position $(s) is inside the CPML region!"
#     end
# end

# @views function check_positions(model::Acoustic_CD_CPML_WaveSimul{3}, positions::Matrix{<:Real})
#     check_positions(ReflectiveBoundaryCondition(), model, positions)
#     for s in 1:size(positions, 1)
#         @assert (model.gridspacing[1] * model.halo <= positions[s, 1] <= model.ls[1] - (model.gridspacing[1] * model.halo) &&
#                  model.gridspacing[2] * model.halo <= positions[s, 2] <= model.ls[2] - (model.gridspacing[2] * model.halo) &&
#                  (model.freetop ? 0 : model.gridspacing[3] * model.halo) <= positions[s, 3] <= model.ls[3] - (model.gridspacing[3] * model.halo)) "Position $(s) is inside the CPML region!"
#     end
# end
