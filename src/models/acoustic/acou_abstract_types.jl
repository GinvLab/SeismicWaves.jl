
abstract type AcousticWaveSimul{N} <: WaveSimul{N} end

abstract type AcousticCDWaveSimul{N} <: AcousticWaveSimul{N} end

abstract type AcousticVDWaveSimul{N} <: AcousticWaveSimul{N} end
