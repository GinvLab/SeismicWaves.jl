### UPDATE MATERIAL PROPERTIES ##

function set_wavesim_matprop!(wavesim::WaveSimulation{T, N}, matprop::MaterialProperties{T, N}) where {T, N}
    @debug "Checking new material properties"
    check_matprop(wavesim, matprop)
    @debug "Updating WaveSimulation material properties"
    update_matprop!(wavesim, matprop)
end