
abstract type WaveSimul{N} end

abstract type AcousticWaveSimul{N} <: WaveSimul{N} end

abstract type AcousticCDWaveSimul{N} <: AcousticWaveSimul{N} end

abstract type InputParameters{N} end

abstract type InputBoundaryConditionParameters end

abstract type MaterialProperties{N} end
