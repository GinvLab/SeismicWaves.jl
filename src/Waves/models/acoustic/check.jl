"""
    @views check_courant_condition(::IsotropicAcoustic, model::WaveModel2D)

Check the Courant number for isotropic acoustic 2D models.
"""
@views function check_courant_condition(::IsotropicAcoustic, model::WaveModel2D)
    vel_max = maximum(model.vel)
    courant = vel_max * model.dt * sqrt(1 / model.dx^2 + 1 / model.dz^2)
    @info "Courant number: $(courant)"
    @assert courant <= 1.0 "Courant condition not satisfied!"
end

"""
    @views check_courant_condition(::IsotropicAcoustic, model::WaveModel3D)

Check the Courant number for isotropic acoustic 3D models.
"""
@views function check_courant_condition(::IsotropicAcoustic, model::WaveModel3D)
    vel_max = maximum(model.vel)
    courant = vel_max * model.dt * sqrt(1 / model.dx^2 + 1 / model.dy^2 + 1 / model.dz^2)
    @info "Courant number: $(courant)"
    @assert courant <= 1.0 "Courant condition not satisfied!"
end