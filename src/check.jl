"""
    check(model::WaveModel)

Check model for possible assertions based on configuration and model traits.
"""
check(model::WaveModel) = check(WaveEquationTrait(model), model)

function check(x::AcousticWaveEquation, model::WaveModel)
    @debug "Checking CFL condition"
    check_courant_condition(x, model)
end
