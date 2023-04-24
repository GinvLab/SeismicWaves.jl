
####################################################

function check_courant_condition(model::AcousticCDWaveSimul, matprop::VpAcousticCDMaterialProperty)
    vel_max = get_maximum_func(model)(matprop.vp)
    tmp = sqrt(sum(1 ./ model.gridspacing .^ 2))
    courant = vel_max * model.dt * tmp
    @debug "Courant number: $(courant)"
    @assert courant <= 1.0 "Courant condition not satisfied! [$(courant)]"
end

####################################################

function check_ppw(
    model::AcousticCDWaveSimul,
    srcs::Sources{<:Real},
    min_ppw::Integer=10
)
    vel_min = get_minimum_func(model)(model.matprop.vp)
    h_max = maximum(model.gridspacing)
    ppw = vel_min / srcs.domfreq / h_max
    @debug "Points per wavelengh: $(ppw)"
    @assert ppw >= min_ppw "Not enough points per wavelengh!"
end

####################################################

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

####################################################

function check_invcov_matrix(model::WaveSimul, invcov)
    @assert size(invcov) == (model.nt, model.nt) "Inverse of covariance matrix has not size equal to ($(model.nt) x $(model.nt))!"
end

#####################################################
