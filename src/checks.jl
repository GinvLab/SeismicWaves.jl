
function check_sim_consistency(model::WaveSimulation{T, N}, matprop::MaterialProperties{T, N}, shots::Vector{Shot{T}}) where {T, N}
    tysource = typeof(shots[1].srcs)
    tyreceiver = typeof(shots[1].recs)

    # Check that the subtypes of Shot are consistent througout the shots
    for singleshot in shots
        if tysource != typeof(singleshot.srcs) || tyreceiver != typeof(singleshot.recs)
            return error("Types of shots are inconsistent.")
        end
    end

    # Check that the subtypes of WaveSimulation, MaterialProperties and Shot are consistent
    if model isa AcousticCDCPMLWaveSimul{T, N, <:AbstractArray{T, N}} &&
       matprop isa VpAcousticCDMaterialProperties{T, N} &&
       tysource <: ScalarSources &&
       tyreceiver <: ScalarReceivers
        return

    elseif model isa AcousticVDStaggeredCPMLWaveSimul{T, N} &&
           matprop isa VpRhoAcousticVDMaterialProperties{T, N} &&
           tysource <: ScalarSources &&
           tyreceiver <: ScalarReceivers
        return

    elseif model isa ElasticIsoCPMLWaveSimul{T, 2} &&   # <<<<<---------<<<<
           matprop isa ElasticIsoMaterialProperties{T, 2} &&
           tysource <: MomentTensorSources &&
           tyreceiver <: VectorReceivers
        return
    end

    return error("Types of WaveSimulation, MaterialProperties and Sources/Receivers are inconsistent \
        \n $(typeof(model)), \n $(typeof(matprop)), \n $(typeof(shots[1].srcs)), $(typeof(shots[1].recs))")
end

function check_shot(model::WaveSimulation, shot::Shot; kwargs...)
    # @debug "Checking model/shot numerics"
    # check_numerics(model, shot; kwargs...)
    # @debug "Checking sources positions"
    # check_positions(model, shot.srcs.positions)
    # @debug "Checking receivers positions"
    # check_positions(model, shot.recs.positions)
    return
end

check_positions(model, positions) = check_positions(BoundaryConditionTrait(model), model, positions)

function check_positions(
    ::ReflectiveBoundaryCondition,
    model::WaveSimulation{T},
    positions::Matrix{T}
) where {T}
    ndimwavsim = length(model.gridspacing)
    @assert size(positions, 2) == ndimwavsim "Positions matrix do not match the dimension of the model!"

    Ndim = size(positions, 2)
    for s in axes(positions, 1)
        for c in 1:Ndim
            @assert (0 <= positions[s, c] <= model.domainextent[c]) "Position $(positions[s,:]) is not inside the grid!"
        end
    end
    return
end

function check_positions(
    ::CPMLBoundaryCondition,
    model::WaveSimulation{T},
    positions::Matrix{T}
) where {T}
    check_positions(ReflectiveBoundaryCondition(), model, positions)
    Ndim = size(positions, 2)
    for s in axes(positions, 1)
        for c in 1:Ndim
            # Check that positions are outside of the CPML region
            if !(c == Ndim && model.freetop)
                @assert (
                    model.gridspacing[c] * model.halo <=
                    positions[s, c] <=
                    model.domainextent[c] - (model.gridspacing[c] * model.halo)
                ) "Position $(positions[s,:]) is inside the CPML region!"
            end
        end
    end
    return
end

check_invcov_matrix(model::WaveSimulation{T}, invcov::AbstractMatrix{T}) where {T} =
    @assert size(invcov) == (model.nt, model.nt) "Inverse of covariance matrix has not size equal to ($(model.nt) x $(model.nt))!"
