"""
Module for generalized wave propagation solvers.
"""
module SeismicWaves

using LinearAlgebra
using SpecialFunctions
using Printf
using ParallelStencil
using Logging
using DocStringExtensions

# main struct for wave simulation
export WaveSimulation
export build_wavesim
# input parameters
export InputParametersAcoustic, InputParametersElastic
# boundary conditions
export CPMLBoundaryConditionParameters, ReflectiveBoundaryConditionParameters
# material properties
export VpAcousticCDMaterialProperties, VpRhoAcousticVDMaterialProperties
export ElasticIsoMaterialProperties
# export sources, receivers and shots
export ScalarShot, MomentTensorShot
export ScalarSources, MomentTensorSources
export MomentTensor2D, MomentTensor3D
export ScalarReceivers, VectorReceivers
# forward, misfit and gradient functions
export swforward!, swmisfit!, swgradient!
# source time functions
export gaussstf, gaussderivstf, rickerstf

include("utils/abstract_types.jl")

# Traits
include("traits/boundarycondition.jl")
include("traits/shooting.jl")
include("traits/snappable.jl")
include("traits/grid.jl")

# Utils
include("utils/utils.jl")
include("utils/checks.jl")
include("utils/fields.jl")
include("utils/grids.jl")
include("utils/checkpointers.jl")
include("utils/snapshotter.jl")

# Shots
include("shots/sources.jl")
include("shots/receivers.jl")
include("shots/shot.jl")

# General models
include("models/bdc_params.jl")
include("models/cpmlcoeffs.jl")

# Acoustic
include("models/acoustic/acou_abstract_types.jl")
include("models/acoustic/acou_material_properties.jl")
include("models/acoustic/acou_params.jl")
include("models/acoustic/acou_models.jl")
include("models/acoustic/acou_forward.jl")
include("models/acoustic/acou_gradient.jl")
include("models/acoustic/acou_init_bc.jl")

# Elastic
include("models/elastic/ela_abstract_types.jl")
include("models/elastic/ela_params.jl")
include("models/elastic/ela_material_properties.jl")
include("models/elastic/ela_models.jl")
include("models/elastic/ela_forward.jl")
#include("models/elastic/ela_gradient.jl")
include("models/elastic/ela_init_bc.jl")

# Backend selection
include("models/backend_selection.jl")

# Inversion 
include("inversion/misfits/L2Misfit.jl")
include("inversion/regularizations/ZerothOrderTikhonovRegularization.jl")

# APIs
include("apis/build.jl")
include("apis/utils.jl")
include("apis/forward.jl")
include("apis/misfit.jl")
include("apis/gradient.jl")


# Acoustic serial backend
include("models/acoustic/backends/Acoustic1D_CD_CPML_Serial.jl")
include("models/acoustic/backends/Acoustic2D_CD_CPML_Serial.jl")
include("models/acoustic/backends/Acoustic3D_CD_CPML_Serial.jl")

# Elastic serial backend
include("models/elastic/backends/Elastic2D_Iso_CPML_Serial.jl")

# Acoustic parallel backends
include("models/acoustic/backends/Acoustic1D_CD_CPML_Threads.jl")
include("models/acoustic/backends/Acoustic2D_CD_CPML_Threads.jl")
include("models/acoustic/backends/Acoustic3D_CD_CPML_Threads.jl")
include("models/acoustic/backends/Acoustic1D_VD_CPML_Threads.jl")
include("models/acoustic/backends/Acoustic2D_VD_CPML_Threads.jl")

# Elastic parallel backends

## HMC stuff
include("HMCseiswaves.jl")
using .HMCseiswaves
export AcouWavCDProb

end # module
