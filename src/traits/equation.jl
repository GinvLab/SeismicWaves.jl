# """
# Abstract trait for a general wave equation model.
# """
# abstract type WaveEquationTrait end

# """
# Trait for an acoustic wave equation model.
# """
# struct Acoustic_CD_WaveSimul <: WaveEquationTrait end

# """
# Abstract trait for a general elastic wave equation model.
# """
# abstract type ElasticWaveEquation <: WaveEquationTrait end

# # Trait constuctor
# WaveEquationTrait(x) = WaveEquationTrait(typeof(x))
# WaveEquationTrait(x::Type) = error("WaveEquationTrait not implemented for type $(x)")
