"""
Abstract trait for a general boundary condition.
"""
abstract type BoundaryConditionTrait end

"""
Trait for a reflective boundary condition.
"""
struct Reflective <: BoundaryConditionTrait end

"""
Trait for a Gaussian taper boundary condition.
"""
struct GaussianTaper <: BoundaryConditionTrait end

"""
Trait for a CPML condition.
"""
struct CPML <: BoundaryConditionTrait end

# Trait constuctor
BoundaryConditionTrait(x) = BoundaryConditionTrait(typeof(x))
BoundaryConditionTrait(x::Type) = error("BoundaryConditionTrait not implemented for type $(x)")