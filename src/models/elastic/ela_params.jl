@doc """
$(TYPEDEF)

Parameters for elastic wave simulations

$(TYPEDFIELDS)
"""
struct InputParametersElastic{T,N} <: InputParameters{T,N}
    "Number of time steps"
    ntimesteps::Int
    "Time step"
    dt::T
    "Grid size for each dimension"
    gridsize::NTuple{N, Int}
    "Grid spacing in each direction"
    gridspacing::NTuple{N, T}
    "Kind of boundary conditions"
    boundcond::InputBoundaryConditionParameters

    function InputParametersElastic(
        ntimesteps::Int,
        dt::T,
        gridsize::NTuple{N, Int},
        gridspacing::NTuple{N, T},
        boundcond::InputBoundaryConditionParameters
    ) where {T,N}
        @assert N <= 3 "Dimensionality must be less than or equal to 3!"
        new{T,N}(ntimesteps, dt, tuple(gridsize...), tuple(gridspacing...), boundcond)
    end
    
end