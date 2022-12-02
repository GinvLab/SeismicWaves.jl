"""
Abstract trait for a general wave equation model.
"""
abstract type WaveEquationTrait end

"""
Abstract trait for a general acoustic wave equation model.
"""
abstract type Acoustic <: WaveEquationTrait end

"""
Trait for an isotropic acoustic wave equation model.
"""
struct IsotropicAcoustic <: WaveEquationTrait end

"""
Abstract trait for a general elastic wave equation model.
"""
abstract type Elastic <: WaveEquationTrait end

# Trait constuctor
WaveEquationTrait(x) = WaveEquationTrait(typeof(x))
WaveEquationTrait(x::Type) = error("WaveEquationTrait not implemented for type $(x)")