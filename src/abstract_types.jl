
@doc """
$(TYPEDEF)

Abstract type for wave simulations.
"""
abstract type WaveSimul{T, N} end

abstract type InputParameters{T, N} end

abstract type InputBoundaryConditionParameters{T} end

abstract type MaterialProperties{T, N} end

abstract type Sources end

abstract type Receivers end

abstract type MomentTensor{T} end

abstract type InterpolationMethod end

abstract type AbstractMisfit end

abstract type AbstractRegularization end
