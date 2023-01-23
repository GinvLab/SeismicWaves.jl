"""
Abstract trait for a general wave equation model.
"""
abstract type WaveEquationTrait end

"""
Abstract trait for a general acoustic wave equation model.
"""
abstract type AcousticWaveEquation <: WaveEquationTrait end

"""
Trait for an isotropic acoustic wave equation model.
"""
struct IsotropicAcousticWaveEquation <: WaveEquationTrait end

"""
Abstract trait for a general elastic wave equation model.
"""
abstract type ElasticWaveEquation <: WaveEquationTrait end

# Trait constuctor
WaveEquationTrait(x) = WaveEquationTrait(typeof(x))
WaveEquationTrait(x::Type) = error("WaveEquationTrait not implemented for type $(x)")