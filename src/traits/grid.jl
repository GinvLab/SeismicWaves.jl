using MPI: MPI

"""
Abstract trait for a general finite difference grid.
"""
abstract type GridTrait end

"""
Trait for a local grid (single node).
"""
struct LocalGrid <: GridTrait end

"""
Trait for a global grid (multi node).
"""
struct GlobalGrid <: GridTrait end

# Trait constuctor
GridTrait(x) = GridTrait(typeof(x))
GridTrait(::Type) = error("GridTrait not implemented for type $(x)")

# Reduce functions

get_maximum_func(model::WaveSimul) = get_maximum_func(GridTrait(model))
get_maximum_func(::LocalGrid) = Base.maximum
get_maximum_func(::GlobalGrid) = (x -> (max_l = maximum(x); MPI.Allreduce(max_l, MPI.MAX, MPI.COMM_WORLD)))

get_minimum_func(model::WaveSimul) = get_minimum_func(GridTrait(model))
get_minimum_func(::LocalGrid) = Base.minimum
get_minimum_func(::GlobalGrid) = (x -> (min_l = minimum(x); MPI.Allreduce(min_l, MPI.MIN, MPI.COMM_WORLD)))
