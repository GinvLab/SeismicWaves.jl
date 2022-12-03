"""
    @views precompute_fact!(::IsotropicAcoustic, model::WaveModel2D)

Precomputes factors for 2D for isotropic acoustic 2D models.
"""
@views function precompute_fact!(::IsotropicAcoustic, model::WaveModel2D)
    model.fact_dx .= model.vel .^ 2 .* (model.dt^2 / model.dx^2)
    model.fact_dz .= model.vel .^ 2 .* (model.dt^2 / model.dz^2)
end

"""
    @views precompute_fact!(::IsotropicAcoustic, model::WaveModel3D)

Precomputes factors for isotropic acoustic 3D models.
"""
@views function precompute_fact!(::IsotropicAcoustic, model::WaveModel3D)
    model.fact_dx .= model.vel .^ 2 .* (model.dt^2 / model.dx^2)
    model.fact_dy .= model.vel .^ 2 .* (model.dt^2 / model.dy^2)
    model.fact_dz .= model.vel .^ 2 .* (model.dt^2 / model.dz^2)
end