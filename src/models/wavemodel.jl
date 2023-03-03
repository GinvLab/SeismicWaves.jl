"""
Abstract type for a general wave propagation model. 
"""
abstract type WaveModel end

"""
Abstract type for a 1D wave propagation model.
"""
abstract type WaveModel1D <: WaveModel end

"""
Abstract type for a 2D wave propagation model.
"""
abstract type WaveModel2D <: WaveModel end

"""
Abstract type for a 3D wave propagation model.
"""
abstract type WaveModel3D <: WaveModel end
