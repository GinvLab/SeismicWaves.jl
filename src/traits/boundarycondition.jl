"""
Abstract trait for a general boundary condition.
"""
abstract type BoundaryConditionTrait end

"""
Trait for a reflective boundary condition.
"""
struct ReflectiveBoundaryCondition <: BoundaryConditionTrait end

"""
Trait for a Gaussian taper boundary condition.
"""
struct GaussianTaperBoundaryCondition <: BoundaryConditionTrait end

"""
Trait for a CPML condition.
"""
struct CPMLBoundaryCondition <: BoundaryConditionTrait end

# Trait constuctor
BoundaryConditionTrait(x) = BoundaryConditionTrait(typeof(x))
BoundaryConditionTrait(::Type) = error("BoundaryConditionTrait not implemented for type $(x)")
