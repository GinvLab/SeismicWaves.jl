"""
Abstract trait for checking if a model is snappable.
"""
abstract type IsSnappableTrait end

"""
Trait for snappable models.
"""
struct Snappable <: IsSnappableTrait end
"""
Trait for unsnappable models.
"""
struct UnSnappable <: IsSnappableTrait end

# Trait constuctor
IsSnappableTrait(x) = IsSnappableTrait(typeof(x))
IsSnappableTrait(::Type) = error("IsSnappableTrait not implemented for type $(x)")

# Default behaviour for a WaveSimulation is UnSnappable
IsSnappableTrait(::Type{<:WaveSimulation}) = UnSnappable()

"""
    snapenabled(model::WaveSimulation)

Check if a model has snapping enabled.
"""
snapenabled(model::WaveSimulation) = isa(IsSnappableTrait(model), Snappable) && model.snapshotter !== nothing
