"""
    check(model::WaveModel)

Check model for possible assertions based on configuration and model traits.
"""
check(model::WaveModel) = check(WaveEquationTrait(model), model)

check(x::IsotropicAcoustic, model::WaveModel) = check_courant_condition(x, model)