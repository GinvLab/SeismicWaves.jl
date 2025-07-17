@doc """
$(TYPEDEF)

Parameters for elastic wave simulations

$(TYPEDFIELDS)
"""
struct InputParametersElastic{T, N} <: InputParameters{T, N}
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

    function InputParametersElastic(
        ntimesteps::Int,
        dt::T,
        gridsize::NTuple{N, Int},
        gridspacing::NTuple{N, T},
        boundcond::InputBoundaryConditionParameters{T}
    ) where {T, N}
        @assert N <= 3 "Dimensionality must be less than or equal to 3!"
        @assert all(gridsize .> 0) "All numbers of grid points must be positive!"
        @assert all(gridspacing .> 0) "All grid spacings must be positive!"
        @assert ntimesteps > 0 "Number of timesteps must be positive!"
        @assert dt > 0 "Timestep size must be positive!"
        new{T, N}(ntimesteps, dt, tuple(gridsize...), tuple(gridspacing...), boundcond)
    end
end
