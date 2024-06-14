struct UniformFiniteDifferenceGrid{N, T} <: AbstractGrid{N, T}
    domainextent::NTuple{N, T}
    ns::NTuple{N, Int}
    gridspacing::NTuple{N, T}
    fields::Dict{String, AbstractField{N,T}}

    function UniformFiniteDifferenceGrid(
        ns::NTuple{N, Int}, gridspacing::NTuple{N, T}
    ) where {N, T}
        # Check numerics
        @assert all(ns .> 0) "All numbers of grid points must be positive!"
        @assert all(gridspacing .> 0) "All cell sizes must be positive!"
        # Compute domain extent
        domainextent = gridspacing .* (ns .- 1)
        new{N, T}(domainextent, ns, gridspacing, Dict())
    end
end

function addfield!(grid::UniformFiniteDifferenceGrid{N, T}, field::Pair{String, <:AbstractField{N,T}}) where {N, T}
    name = field.first
    if !haskey(grid.fields, name)
        grid.fields[name] = field.second
    else
        error("Field with name [$(name)] already in the grid!")
    end
end

function reset!(grid::UniformFiniteDifferenceGrid{N,T}; except::Vector{String}=[]) where {N,T}
    for (name, field) in grid.fields
        if !(name in except)
            copyto!(field, ScalarConstantField{N,T}(0.0))
        end
    end
end