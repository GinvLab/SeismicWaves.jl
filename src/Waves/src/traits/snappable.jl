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

# Default behaviour for a WaveModel is UnSnappable
IsSnappableTrait(::Type{<:WaveModel}) = UnSnappable()

"""
    snapenabled(model::WaveModel)

Check if a model has snapping enabled.
"""
snapenabled(model::WaveModel) = isa(IsSnappableTrait(model), Snappable) && model.snapevery !== nothing
