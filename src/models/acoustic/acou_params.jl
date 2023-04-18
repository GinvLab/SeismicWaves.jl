"""
Parameters for acoustic wave simulations
"""
struct InputParametersAcoustic{N} <: InputParameters{N}
    ntimesteps::Int
    dt::Real
    gridsize::NTuple{N, <:Int}
    gridspacing::NTuple{N, <:Real}
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

"""
Reflective boundary conditions parameters for acoustic wave simulations
"""
Base.@kwdef struct ReflectiveBoundaryConditionParameters <: InputBoundaryConditionParameters end

"""
CPML boundary conditions parameters for acoustic wave simulations
"""
Base.@kwdef struct CPMLBoundaryConditionParameters <: InputBoundaryConditionParameters
    halo::Int = 20
    rcoef::Real = 0.0001
    freeboundtop::Bool = true
end
