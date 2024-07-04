struct UniformFiniteDifferenceGrid{N, T} <: AbstractGrid{N, T}
    extent::NTuple{N, T}
    size::NTuple{N, Int}
    spacing::NTuple{N, T}
    fields::Dict{String, AbstractField{T}}

    function UniformFiniteDifferenceGrid(
        size::NTuple{N, Int}, spacing::NTuple{N, T}
    ) where {N, T}
        # Check numerics
        @assert all(size .> 0) "All numbers of grid points must be positive!"
        @assert all(spacing .> 0) "All cell sizes must be positive!"
        # Compute domain extent
        extent = spacing .* (size .- 1)
        new{N, T}(extent, size, spacing, Dict())
    end
end

function addfield!(grid::UniformFiniteDifferenceGrid{N, T}, field::Pair{String, <:AbstractField{T}}) where {N, T}
    name = field.first
    if !haskey(grid.fields, name)
        grid.fields[name] = field.second
    else
        error("Field with name [$(name)] already in the grid!")
    end
end

reset!(grid::UniformFiniteDifferenceGrid{N, T}; except::Vector{String}=[]) where {N, T} =
    for (name, field) in grid.fields
        if !(name in except)
            setzero!(field)
        end
    end