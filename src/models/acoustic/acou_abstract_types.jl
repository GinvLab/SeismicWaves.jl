
abstract type AcousticWaveSimul{N} <: WaveSimul{N} end

abstract type AcousticCDWaveSimul{N} <: AcousticWaveSimul{N} end

abstract type AcousticVDStaggeredWaveSimul{N} <: AcousticWaveSimul{N} end
