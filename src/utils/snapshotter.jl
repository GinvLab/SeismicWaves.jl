struct LinearSnapshotter{T, N, A} <: AbstractSnapshotter{T, N}
    nt::Int
    snapevery::Int
    snapshots::Dict{Int, Dict{String, <:AbstractField{T}}}

    function LinearSnapshotter{A}(
        nt::Int,
        snapevery::Int,
        snapshotted_fields::Dict{String, <:AbstractField{T}}
    ) where {T, N, A <: AbstractArray{T, N}}
        @assert snapevery < nt "Checkpointing frequency must be smaller than the number of timesteps!"
        # Preallocate snapshots
        snapshots = Dict{Int, Dict{String, Union{ScalarVariableField{T, N, A}, MultiVariableField{T, N, A}}}}()
        for it in 1:nt
            if it % snapevery == 0
                for (name, field) in snapshotted_fields
                    if !haskey(snapshots, it)
                        snapshots[it] = Dict()
                    end
                    if typeof(field) <: ScalarVariableField
                        cfield = copy(convert(A, field.value))
                        snapshots[it][name] = ScalarVariableField(cfield)
                    elseif typeof(field) <: MultiVariableField
                        cfields = [copy(convert(A, field.value[i])) for i in 1:length(field.value)]
                        snapshots[it][name] = MultiVariableField(cfields)
                    end
                end
            end
        end
        # Preallocate buffers
        new{T, N, A}(nt, snapevery, snapshots)
    end
end

function savesnapshot!(snapshotter::LinearSnapshotter{T, N}, field::Pair{String, <:AbstractField{T}}, it::Int) where {T, N}
    # Save field in snapshots
    if it % snapshotter.snapevery == 0
        @info @sprintf("Snapping iteration: %d", it)
        copyto!(snapshotter.snapshots[it][field.first], field.second)
    end
end
