"""
    @views precompute_fact!(::IsotropicAcousticWaveEquation, model::WaveModel)

Precomputes factors for isotropic acoustic models.
"""
@views function precompute_fact!(::IsotropicAcousticWaveEquation, model::WaveModel)
    model.fact .= (model.dt^2) .* (model.vel .^ 2)
end