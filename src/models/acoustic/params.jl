"""
  Parameters for acoustic wave simulations
"""
struct InputParametersAcoustic{N} <: InputParameters{N}
    ntimesteps::Int
    dt::Real
    ns::NTuple{N,<:Int}
    Δs::NTuple{N,<:Real}
    boundcond::InputBDCParameters
end

function InputParametersAcoustic(
    ntimesteps::Int,
    dt::Real,
    ns::AbstractVector{<:Int},
    Δs::AbstractVector{<:Real},
    boundcond::InputBDCParameters
)
    # Check dimensionality
    N = length(ns)
    @assert N == length(Δs) "Dimensionality mismatch between number of grid points and grid step sizes!"
    @assert N > 0 "Dimensionality must positive!"
    @assert N <= 3 "Dimensionality must be less than or equal to 3!"

    InputParametersAcoustic{N}(ntimesteps, dt, tuple(ns...), tuple(Δs...), boundcond)
end

"""
  Reflective boundary conditions parameters for acoustic wave simulations
"""
Base.@kwdef struct InputBDCParametersAcousticReflective <: InputBDCParameters end

"""
  CPML boundary conditions parameters for acoustic wave simulations
"""
Base.@kwdef struct InputBDCParametersAcousticCPML <: InputBDCParameters
    halo::Int = 20
    rcoef::Real = 0.0001
    freeboundtop::Bool = false
end
