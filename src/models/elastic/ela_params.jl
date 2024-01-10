@doc """
$(TYPEDEF)

Parameters for elastic wave simulations

$(TYPEDFIELDS)
"""
struct InputParametersElastic{N} <: InputParameters{N}
    "Number of time steps"
    ntimesteps::Int
    "Time step"
    dt::Real
    "Grid size for each dimension"
    gridsize::NTuple{N, <:Int}
    "Grid spacing in each direction"
    gridspacing::NTuple{N, <:Real}
    "Kind of boundary conditions"
    boundcond::InputBoundaryConditionParameters
end

function InputParametersElastic(
    ntimesteps::Int,
    dt::Real,
    gridsize::AbstractVector{<:Int},
    gridspacing::AbstractVector{<:Real},
    boundcond::InputBoundaryConditionParameters
)
    # Check dimensionality
    N = length(gridsize)
    @assert N == length(gridspacing) "Dimensionality mismatch between number of grid points and grid step sizes!"
    @assert N > 0 "Dimensionality must positive!"
    @assert N <= 3 "Dimensionality must be less than or equal to 3!"

    return InputParametersElastic{N}(ntimesteps, dt, tuple(gridsize...), tuple(gridspacing...), boundcond)
end

