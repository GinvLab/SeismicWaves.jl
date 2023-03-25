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
IsSnappableTrait(x::Type) = error("IsSnappableTrait not implemented for type $(x)")

# Default behaviour for a WaveSimul is UnSnappable
IsSnappableTrait(::Type{<:WaveSimul}) = UnSnappable()

"""
    snapenabled(model::WaveSimul)

Check if a model has snapping enabled.
"""
snapenabled(model::WaveSimul) = isa(IsSnappableTrait(model), Snappable) && model.snapevery !== nothing
