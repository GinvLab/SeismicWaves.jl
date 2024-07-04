abstract type WaveSimulation{T, N} end

abstract type InputParameters{T, N} end

abstract type InputBoundaryConditionParameters{T} end

abstract type MaterialProperties{T, N} end

abstract type Shot{T} end

abstract type Sources{T} end

abstract type Receivers{T} end

abstract type MomentTensor{T, N} end

abstract type InterpolationMethod end

abstract type AbstractMisfit end

abstract type AbstractRegularization end

abstract type AbstractField{T} end

abstract type AbstractGrid{N, T} end

abstract type AbstractCheckpointer{T} end

abstract type AbstractSnapshotter{T, N} end