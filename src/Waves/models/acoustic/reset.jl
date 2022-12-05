"""
    @views reset!(::IsotropicAcoustic, model::WaveModel)

Resets pressure fields for an isotropic acoustic wave model.
"""
@views function reset_pressure!(::IsotropicAcoustic, model::WaveModel)
    fill!(model.pold, 0)
    fill!(model.pcur, 0)
    fill!(model.pnew, 0)
end