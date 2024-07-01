
@doc """
$(TYPEDEF)

Abstract type for wave simulations.
"""
abstract type WaveSimul{N} end

abstract type InputParameters{N} end

abstract type InputBoundaryConditionParameters end

abstract type MaterialProperties{T, N} end

abstract type Sources end

abstract type Receivers end

abstract type MomentTensor end

abstract type InterpolationMethod end

abstract type AbstractMisfit end

abstract type AbstractRegularization end
