mutable struct LinearCheckpointer{T} <: AbstractCheckpointer{T}
    nt::Int
    check_freq::Int
    last_checkpoint::Int
    curr_checkpoint::Int
    buffers::Dict{String, Vector{<:AbstractField{T}}}
    checkpoints::Dict{Int, Dict{String, <:AbstractField{T}}}
    widths::Dict{String, Int}

    function LinearCheckpointer(
        nt::Int,
        check_freq::Int,
        checkpointed_fields::Dict{String, <:AbstractField{T}},
        buffered_fields::Vector{String};
        widths::Dict{String, Int}=Dict()
    ) where {T}
        @assert check_freq < nt "Checkpointing frequency must be smaller than the number of timesteps!"
        # Time step of last checkpoint
        last_checkpoint = floor(Int, nt / check_freq) * check_freq
        # Preallocate checkpoints
        checkpoints = Dict{Int, Dict{String, AbstractField{T}}}()
        for it in 0:(nt+1)
            if it % check_freq == 0
                for (name, field) in checkpointed_fields
                    w = haskey(widths, name) ? widths[name] : 1
                    for itw in it:-1:it-w+1
                        if !haskey(checkpoints, itw)
                            checkpoints[itw] = Dict()
                        end
                        checkpoints[itw][name] = zero(field)
                    end
                end
            end
        end
        # Preallocate buffers
        buffers = Dict{String, Vector{AbstractField{T}}}()
        for name in buffered_fields
            buffers[name] = [zero(checkpointed_fields[name]) for _ in 1:check_freq+1]
        end
        new{T}(nt, check_freq, last_checkpoint, last_checkpoint, buffers, checkpoints, widths)
    end
end

function savecheckpoint!(checkpointer::LinearCheckpointer{T}, field::Pair{String, <:AbstractField{T}}, it::Int) where {T}
    # Save field in checkpoints
    w = haskey(checkpointer.widths, field.first) ? checkpointer.widths[field.first] : 1
    for itw in it:it+w-1
        if itw % checkpointer.check_freq == 0
            copyto!(checkpointer.checkpoints[it][field.first], field.second)
        end
    end
    # Start populating buffer at last checkpoint
    if haskey(checkpointer.buffers, field.first) && it >= checkpointer.last_checkpoint
        copyto!(checkpointer.buffers[field.first][it-checkpointer.last_checkpoint+1], field.second)
    end
end

function isbuffered(checkpointer::LinearCheckpointer, field::String, it::Int)::Bool
    return haskey(checkpointer.buffers, field) && (checkpointer.curr_checkpoint <= it <= checkpointer.curr_checkpoint + checkpointer.check_freq)
end

function getbuffered(checkpointer::LinearCheckpointer, field::String, it::Int)::AbstractField
    @assert isbuffered(checkpointer, field, it)
    return checkpointer.buffers[field][it-checkpointer.curr_checkpoint+1]
end

function ischeckpointed(checkpointer::LinearCheckpointer, field::String, it::Int)::Bool
    return haskey(checkpointer.checkpoints, it) && haskey(checkpointer.checkpoints[it], field)
end

function getcheckpointed(checkpointer::LinearCheckpointer{T}, field::String, it::Int)::AbstractField{T} where {T}
    @assert ischeckpointed(checkpointer, field, it)
    return checkpointer.checkpoints[it][field]
end

function issaved(checkpointer::LinearCheckpointer, field::String, it::Int)::Bool
    return isbuffered(checkpointer, field, it) || ischeckpointed(checkpointer, field, it)
end

function getsaved(checkpointer::LinearCheckpointer{T}, field::String, it::Int)::AbstractField{T} where {T}
    if ischeckpointed(checkpointer, field, it)
        return getcheckpointed(checkpointer, field, it)
    end
    return getbuffered(checkpointer, field, it)
end

function initrecover!(checkpointer::LinearCheckpointer)
    old_checkpoint = checkpointer.curr_checkpoint
    checkpointer.curr_checkpoint -= checkpointer.check_freq
    for (name, buffer) in checkpointer.buffers
        copyto!(buffer[1], checkpointer.checkpoints[checkpointer.curr_checkpoint][name])
        copyto!(buffer[end], checkpointer.checkpoints[old_checkpoint][name])
    end
end

function recover!(checkpointer::LinearCheckpointer, recoverfun)
    start_rec_it = checkpointer.curr_checkpoint + 1
    end_rec_it = checkpointer.curr_checkpoint + checkpointer.check_freq - 1
    for it in start_rec_it:end_rec_it
        recovered_fields = recoverfun(it)
        for (name, field) in recovered_fields
            copyto!(checkpointer.buffers[name][it-start_rec_it+2], field)
        end
    end
end

reset!(checkpointer::LinearCheckpointer) = checkpointer.curr_checkpoint = checkpointer.last_checkpoint