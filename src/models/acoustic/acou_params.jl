
@doc """
$(TYPEDEF)

Parameters for acoustic wave simulations.

$(TYPEDFIELDS)
"""
struct InputParametersAcoustic{N} <: InputParameters{N}
    "Number of time steps"
    ntimesteps::Int
    "Time step"
    dt::Real
    "Grid for each dimension"
    gridsize::NTuple{N, <:Int}
    "Grid spacing in each direction"
    gridspacing::NTuple{N, <:Real}
    "Kind of boundary conditions"
    boundcond::InputBoundaryConditionParameters
end

function InputParametersAcoustic(
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

    return InputParametersAcoustic{N}(ntimesteps, dt, tuple(gridsize...), tuple(gridspacing...), boundcond)
end

@doc """
$(TYPEDEF)

Parameters for variable-density acoustic wave simulations.

$(TYPEDFIELDS)
"""
struct InputParametersAcousticVariableDensity{N} <: InputParameters{N}
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

function InputParametersAcousticVariableDensity(
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

    return InputParametersAcousticVariableDensity{N}(ntimesteps, dt, tuple(gridsize...), tuple(gridspacing...), boundcond)
end

@doc """
$(TYPEDEF)

Reflective boundary conditions parameters for acoustic wave simulations.
"""
Base.@kwdef struct ReflectiveBoundaryConditionParameters <: InputBoundaryConditionParameters end

@doc """
$(TYPEDEF)

CPML boundary conditions parameters for acoustic wave simulations.

$(TYPEDFIELDS)
"""
Base.@kwdef struct CPMLBoundaryConditionParameters <: InputBoundaryConditionParameters
    "Number of CPML grid points"
    halo::Int = 20
    "Target reflection coefficient"
    rcoef::Real = 0.0001
    "Free surface boundary condition at the top"
    freeboundtop::Bool = true
end
