
@doc """
$(TYPEDEF)

Reflective boundary conditions parameters for wave simulations.
"""
Base.@kwdef struct ReflectiveBoundaryConditionParameters{T, N} <: InputBoundaryConditionParameters{T} end

@doc """
$(TYPEDEF)

CPML boundary conditions parameters for wave simulations.

$(TYPEDFIELDS)
"""
Base.@kwdef struct CPMLBoundaryConditionParameters{T} <: InputBoundaryConditionParameters{T}
    "Number of CPML grid points"
    halo::Int = 20
    "Target reflection coefficient"
    rcoef::T = 0.0001
    "Free surface boundary condition at the top"
    freeboundtop::Bool = true
end
