
function check(model::WaveSimul, matprop::MaterialProperties)
    @debug "Checking CFL condition"
    return check_courant_condition(model, matprop)
end
