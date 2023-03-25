"""
  Parameters for acoustic wave simulations
"""
struct InputParametersAcoustic{N} <: InputParameters{N}
    ntimesteps::Int
    dt::Real
    gridsize::NTuple{N,<:Int}
    gridspacing::NTuple{N,<:Real}
    boundcond::InputBCParameters
end

function InputParametersAcoustic(
    ntimesteps::Int,
    dt::Real,
    gridsize::AbstractVector{<:Int},
    gridspacing::AbstractVector{<:Real},
    boundcond::InputBCParameters
    )
    # Check dimensionality
    N = length(gridsize)
    @assert N == length(gridspacing) "Dimensionality mismatch between number of grid points and grid step sizes!"
    @assert N > 0 "Dimensionality must positive!"
    @assert N <= 3 "Dimensionality must be less than or equal to 3!"

    InputParametersAcoustic{N}(ntimesteps, dt, tuple(gridsize...), tuple(gridspacing...), boundcond)
end

"""
  Reflective boundary conditions parameters for acoustic wave simulations
"""
#Base.@kwdef struct InputBCParametersAcousticReflective <: InputBCParameters end
Base.@kwdef struct Refl_BC <: InputBCParameters end

"""
  CPML boundary conditions parameters for acoustic wave simulations
"""
#Base.@kwdef struct CPML_BC <: InputBCParameters
Base.@kwdef struct CPML_BC <: InputBCParameters
    halo::Int = 20
    rcoef::Real = 0.0001
    freeboundtop::Bool = false
end
