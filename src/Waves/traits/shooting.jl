"""
Abstract trait for a general multishooting model.
"""
abstract type ShootingTrait end

"""
Trait for sequential shooting model.
"""
struct SequentialShooting <: ShootingTrait end

"""
Trait for distributed shooting model.
"""
struct DistributedShooting <: ShootingTrait end

# Trait constuctor
ShootingTrait(x) = ShootingTrait(typeof(x))
ShootingTrait(x::Type) = error("ShootingTrait not implemented for type $(x)")

# Default constructor for a WaveModel
ShootingTrait(::Type{<:WaveModel}) = SequentialShooting()