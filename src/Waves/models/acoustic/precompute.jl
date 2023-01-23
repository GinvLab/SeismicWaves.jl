"""
    @views precompute_fact!(::IsotropicAcousticWaveEquation, model::WaveModel2D)

Precomputes factors for 2D for isotropic acoustic 2D models.
"""
@views function precompute_fact!(::IsotropicAcousticWaveEquation, model::WaveModel2D)
    model.fact_x .= model.vel .^ 2 .* (model.dt^2 / model.dx^2)
    model.fact_z .= model.vel .^ 2 .* (model.dt^2 / model.dz^2)
end

"""
    @views precompute_fact!(::IsotropicAcousticWaveEquation, model::WaveModel3D)

Precomputes factors for isotropic acoustic 3D models.
"""
@views function precompute_fact!(::IsotropicAcousticWaveEquation, model::WaveModel3D)
    model.fact_x .= model.vel .^ 2 .* (model.dt^2 / model.dx^2)
    model.fact_y .= model.vel .^ 2 .* (model.dt^2 / model.dy^2)
    model.fact_z .= model.vel .^ 2 .* (model.dt^2 / model.dz^2)
end