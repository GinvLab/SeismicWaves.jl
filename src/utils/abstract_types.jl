abstract type WaveSimulation{T, N} end

@doc """

$(TYPEDEF)

`InputParameters` is the abstract supertype describing input parameters for wave simulations.

Currently implemented concrete parameters are [`InputParametersAcoustic`](@ref) and [`InputParametersElastic`](@ref), for acoustic and elasic wave simulation respectively.

"""
abstract type InputParameters{T, N} end

@doc """

$(TYPEDEF)

`InputBoundaryConditionParameters` is the abstract supertype describing boundary conditions input parameters for wave simulations.

Currently implemented concrete parameters are [`CPMLBoundaryConditionParameters`](@ref) and [`ReflectiveBoundaryConditionParameters`](@ref).

"""
abstract type InputBoundaryConditionParameters{T} end

@doc """

$(TYPEDEF)

`MaterialProperties` is the abstract supertype describing material properties for wave simulations. It defines which type of wave equation is solved.

Currently implemented concrete properties are:
- [`VpAcousticCDMaterialProperties`](@ref) for acoustic constant density wave equation
- [`VpRhoAcousticVDMaterialProperties`](@ref) for acoustic variable density wave equation
- [`ElasticIsoMaterialProperties`](@ref) for elasitic isotropic wave equation

"""
abstract type MaterialProperties{T, N} end

@doc """

$(TYPEDEF)

`Shot` is the abstract supertype describing a shot composed of sources and receivers.

Currently implemented concrete shot types are:
- [`ScalarShot`](@ref) for scalar field sources and receivers
- [`MomentTensorShot`](@ref) for moment tensor sources and vector field receivers

"""
abstract type Shot{T} end

@doc """

$(TYPEDEF)

`Sources` is the abstract supertype describing seismic sources.

Currently implemented concrete sources types are:
- [`ScalarSources`](@ref) for scalar field sources
- [`MomentTensorSources`](@ref) for moment tensor sources

"""
abstract type Sources{T} end

@doc """

$(TYPEDEF)

`Sources` is the abstract supertype describing seismic receivers.

Currently implemented concrete receivers types are:
- [`ScalarReceivers`](@ref) for scalar field receivers
- [`VectorReceivers`](@ref) for vector field receivers

"""
abstract type Receivers{T} end

abstract type MomentTensor{T, N} end

abstract type AbstractInterpolationMethod end

@doc """

$(TYPEDEF)

An abstract type to represent misfit functions and related parameters.

"""
abstract type AbstractMisfit{T} end

abstract type AbstractRegularization end

abstract type AbstractField{T} end

abstract type AbstractGrid{N, T} end

abstract type AbstractCheckpointer{T} end

abstract type AbstractSnapshotter{T, N} end
