
abstract type AcousticWaveSimulation{T, N} <: WaveSimulation{T, N} end

abstract type AcousticCDWaveSimulation{T, N} <: AcousticWaveSimulation{T, N} end

abstract type AcousticVDStaggeredWaveSimulation{T, N} <: AcousticWaveSimulation{T, N} end
