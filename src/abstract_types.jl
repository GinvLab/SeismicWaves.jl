
abstract type WaveSimul{N} end

abstract type AcousticWaveSimul{N} <: WaveSimul{N} end

abstract type AcousticCDWaveSimul{N} <: AcousticWaveSimul{N} end

abstract type InputParameters{N} end

abstract type InputBoundaryConditionParameters end

abstract type MaterialProperties{N} end

@doc raw"""
TODO
"""
function set_wavesim_matprop(wavesim::WaveSimul{N}, matprop::MaterialProperties{N}) where {N}
    @debug "Checking new material properties"
    check_matprop!(wavesim, matprop)
    @debug "Updating new material properties"
    update_matprop!(wavesim, matprop)
end