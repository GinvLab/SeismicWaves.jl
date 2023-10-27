"""
Module for generalized wave propagation solvers.
"""
module SeismicWaves

using LinearAlgebra
using Printf
using ParallelStencil
using Logging

export InputParametersAcoustic, InputParametersAcousticVariableDensity
export CPMLBoundaryConditionParameters, ReflectiveBoundaryConditionParameters
export VpAcousticCDMaterialProperty, VpRhoAcousticVDMaterialProperty
export WaveSimul

#export Sources, Receivers, Shot
export ScalarSources, ScalarReceivers, Shot
export swforward!, swmisfit!, swgradient!
export build_wavesim
export gaussource1D, gaussdersource1D, rickersource1D



include("abstract_types.jl")

include("traits/boundarycondition.jl")
include("traits/shooting.jl")
include("traits/snappable.jl")
include("traits/grid.jl")

include("sources.jl")
include("receivers.jl")
include("shot.jl")
include("checks.jl")

include("models/cpmlcoeffs.jl")

include("models/acoustic/acou_abstract_types.jl")
include("models/acoustic/material_properties.jl")
include("models/acoustic/acou_models.jl")
include("models/acoustic/acou_params.jl")
include("models/acoustic/acou_forward.jl")
include("models/acoustic/acou_gradient.jl")
include("models/acoustic/acou_init_bc.jl")

include("misfits.jl")
include("solve.jl")
include("wrappers.jl")

include("models/acoustic/backends/Acoustic1D_CD_CPML_Serial.jl")
include("models/acoustic/backends/Acoustic2D_CD_CPML_Serial.jl")
include("models/acoustic/backends/Acoustic3D_CD_CPML_Serial.jl")

ParallelStencil.@reset_parallel_stencil()
include("models/acoustic/backends/Acoustic1D_CD_CPML_Threads.jl")
ParallelStencil.@reset_parallel_stencil()
include("models/acoustic/backends/Acoustic1D_CD_CPML_GPU.jl")
ParallelStencil.@reset_parallel_stencil()
include("models/acoustic/backends/Acoustic2D_CD_CPML_Threads.jl")
ParallelStencil.@reset_parallel_stencil()
include("models/acoustic/backends/Acoustic2D_CD_CPML_GPU.jl")
ParallelStencil.@reset_parallel_stencil()
include("models/acoustic/backends/Acoustic3D_CD_CPML_Threads.jl")
ParallelStencil.@reset_parallel_stencil()
include("models/acoustic/backends/Acoustic3D_CD_CPML_GPU.jl")

ParallelStencil.@reset_parallel_stencil()
include("models/acoustic/backends/Acoustic1D_VD_CPML_Threads.jl")
ParallelStencil.@reset_parallel_stencil()
include("models/acoustic/backends/Acoustic1D_VD_CPML_GPU.jl")
ParallelStencil.@reset_parallel_stencil()
include("models/acoustic/backends/Acoustic2D_VD_CPML_Threads.jl")
ParallelStencil.@reset_parallel_stencil()
include("models/acoustic/backends/Acoustic2D_VD_CPML_GPU.jl")

include("utils.jl")


## HMC stuff
include("HMCseiswaves.jl")
using .HMCseiswaves
export AcouWavCDProb


end # module
