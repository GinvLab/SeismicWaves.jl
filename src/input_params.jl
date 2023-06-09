
"""
Reflective boundary conditions parameters for wave simulations
"""
Base.@kwdef struct ReflectiveBoundaryConditionParameters <: InputBoundaryConditionParameters end

"""
CPML boundary conditions parameters for wave simulations
"""
Base.@kwdef struct CPMLBoundaryConditionParameters <: InputBoundaryConditionParameters
    halo::Int = 20
    rcoef::Real = 0.0001
    freeboundtop::Bool = true
end
