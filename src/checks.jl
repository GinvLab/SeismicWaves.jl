
function check_shot(model::WaveSimul, shot::Shot; kwargs...)
    @debug "Checking model/shot numerics"
    check_numerics(model, shot; kwargs...)
    @debug "Checking sources positions"
    check_positions(model, shot.srcs.positions)
    @debug "Checking receivers positions"
    return check_positions(model, shot.recs.positions)
end

check_positions(model::WaveSimul, positions::Matrix{<:Real}) = check_positions(BoundaryConditionTrait(model), model, positions)

function check_positions(
    ::ReflectiveBoundaryCondition,
    model::WaveSimul,
    positions::Matrix{<:Real}
)
    ndimwavsim = length(model.gridspacing)
    @assert size(positions, 2) == ndimwavsim "Positions matrix do not match the dimension of the model!"

    Ndim = size(positions, 2)
    for s in axes(positions, 1)
        for c in 1:Ndim
            @assert (0 <= positions[s, c] <= model.ls[c]) "Position $(positions[s,:]) is not inside the grid!"
        end
    end
end

function check_positions(
    ::CPMLBoundaryCondition,
    model::WaveSimul,
    positions::Matrix{<:Real}
)
    check_positions(ReflectiveBoundaryCondition(), model, positions)
    Ndim = size(positions, 2)
    for s in axes(positions, 1)
        for c in 1:Ndim
            # Check that positions are outside of the CPML region
            if !(c == Ndim && model.freetop)
                @assert (
                    model.gridspacing[c] * model.halo <=
                    positions[s, c] <=
                    model.ls[c] - (model.gridspacing[c] * model.halo)
                ) "Position $(positions[s,:]) is inside the CPML region!"
            end
        end
    end
end

function check_invcov_matrix(model::WaveSimul, invcov)
    @assert size(invcov) == (model.nt, model.nt) "Inverse of covariance matrix has not size equal to ($(model.nt) x $(model.nt))!"
end