"""
Module for generalized wave propagation solvers.
"""
module SeismicWaves

using LinearAlgebra
using Printf
using ParallelStencil
using Logging
using DocStringExtensions

# main struct for wave simulation
export WaveSimul
export build_wavesim
# input parameters
export InputParametersAcoustic, InputParametersAcousticVariableDensity
# boundary conditions
export CPMLBoundaryConditionParameters, ReflectiveBoundaryConditionParameters
# material properties
export VpAcousticCDMaterialProperty, VpRhoAcousticVDMaterialProperty
# export sources, receivers and shots
export Shot
export ScalarSources, MomentTensorSources
export ScalarReceivers, VectorReceivers
# forward, misfit and gradient functions
export swforward!, swmisfit!, swgradient!
# source time functions
export gaussstf, gaussderivstf, rickerstf


include("abstract_types.jl")

include("traits/boundarycondition.jl")
include("traits/shooting.jl")
include("traits/snappable.jl")
include("traits/grid.jl")

include("sources.jl")
include("receivers.jl")
include("shot.jl")
include("checks.jl")
include("boundarycond.jl")

include("models/cpmlcoeffs.jl")

# Acoustic
include("models/acoustic/acou_abstract_types.jl")
include("models/acoustic/acou_material_properties.jl")
include("models/acoustic/acou_models.jl")
include("models/acoustic/acou_params.jl")
include("models/acoustic/acou_forward.jl")
include("models/acoustic/acou_gradient.jl")
include("models/acoustic/acou_init_bc.jl")

# Elastic
include("models/elastic/ela_abstract_types.jl")
include("models/elastic/ela_material_properties.jl")
include("models/elastic/ela_models.jl")
include("models/elastic/ela_params.jl")
include("models/elastic/ela_forward.jl")
#include("models/elastic/ela_gradient.jl")
include("models/elastic/ela_init_bc.jl")


# Backend selection
include("models/backend_selection.jl")
# Misfit 
include("misfits.jl")
# Generic solve
include("solve.jl")
# Main functions
include("wrappers.jl")

# Acoustic backends
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


# Elastic backends
include("models/acoustic/backends/Elastic2D_Iso_CPML_Serial.jl")


# Utils
include("utils.jl")


## HMC stuff
include("HMCseiswaves.jl")
using .HMCseiswaves
export AcouWavCDProb


end # module
