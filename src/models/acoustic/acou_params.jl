
@doc """
$(TYPEDEF)

Parameters for acoustic wave simulations.

$(TYPEDFIELDS)
"""
struct InputParametersAcoustic{T, N} <: InputParameters{T, N}
    "Number of time steps"
    ntimesteps::Int
    "Time step"
    dt::T
    "Grid size for each dimension"
    gridsize::NTuple{N, Int}
    "Grid spacing in each direction"
    gridspacing::NTuple{N, T}
    "Kind of boundary conditions"
    boundcond::InputBoundaryConditionParameters{T}

    function InputParametersAcoustic(
        ntimesteps::Int,
        dt::T,
        gridsize::NTuple{N, Int},
        gridspacing::NTuple{N, T},
        boundcond::InputBoundaryConditionParameters{T}
    ) where {T, N}
        @assert N <= 3 "Dimensionality must be less than or equal to 3!"
        new{T, N}(ntimesteps, dt, gridsize, gridspacing, boundcond)
    end
end
