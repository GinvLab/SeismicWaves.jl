

@doc """
$(TYPEDEF)

Reflective boundary conditions parameters for wave simulations.
"""
Base.@kwdef struct ReflectiveBoundaryConditionParameters <: InputBoundaryConditionParameters end

@doc """
$(TYPEDEF)

CPML boundary conditions parameters for wave simulations.

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
