"""
    precompute!(model::WaveModel)

Precompute constant values needed for efficient kernels.
"""
precompute!(model::WaveModel) = precompute!(WaveEquationTrait(model), model)

precompute!(x::AcousticWaveEquation, model::WaveModel) = precompute_fact!(x, model)