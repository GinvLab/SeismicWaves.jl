
abstract type AcousticWaveSimul{T,N} <: WaveSimul{T,N} end

abstract type AcousticCDWaveSimul{T,N} <: AcousticWaveSimul{T,N} end

abstract type AcousticVDStaggeredWaveSimul{T,N} <: AcousticWaveSimul{T,N} end
