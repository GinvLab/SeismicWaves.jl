module SeismicWaves

using LinearAlgebra
using SpecialFunctions
using Printf
using ParallelStencil
using Logging
using DocStringExtensions
using Interpolations
using REPL
using StaticArrays
using DSP
using NumericalIntegration

# main struct for wave simulation
export WaveSimulation
# input parameters
export InputParameters, InputParametersAcoustic, InputParametersElastic
export RunParameters,GradParameters
# boundary conditions
export InputBoundaryConditionParameters
export CPMLBoundaryConditionParameters, ReflectiveBoundaryConditionParameters
# material properties
export MaterialProperties
export VpAcousticCDMaterialProperties, VpRhoAcousticVDMaterialProperties
export ElasticIsoMaterialProperties
# export sources, receivers and shots
export Shot, ScalarShot, MomentTensorShot, ExternalForceShot, PSDMomentTensorShot, PSDExternalForceShot
export Sources, ScalarSources, MomentTensorSources, ExternalForceSources, PSDMomentTensorSources, PSDExternalForceSources
export MomentTensor2D, MomentTensor3D
export Receivers, ScalarReceivers, VectorReceivers, VectorCrossCorrelationsReceivers
# forward, misfit and gradient functions
export build_wavesim, swforward!, swmisfit!, swgradient!
# misfits
export AbstractMisfit, L2Misfit, CCTSMisfit
# source time functions
export gaussstf, gaussderivstf, rickerstf


module FiniteDifferencesMacros
    using MacroTools
    include("utils/fdgen.jl")
    export @∂, @∂², @∂ⁿ
    export @∂x, @∂y, @∂z
    export @∂²x, @∂²y, @∂²z
    export @∂ⁿx, @∂ⁿy, @∂ⁿz
    export @∇, @div, @∇²
    export @∂̃, @∂̃x, @∂̃y, @∂̃z
    export @∂̃², @∂̃²x, @∂̃²y, @∂̃²z
    export @∇̃, @diṽ, @∇̃²
end

module FDGeneratedFunctions
    using StaticArrays
    include("utils/fdgenerated.jl")
    export ∂x4th, ∂y4th, ∂z4th
    export ∂̃x4th, ∂̃y4th, ∂̃z4th
    export ∂²x4th, ∂²y4th, ∂²z4th
    export ∂̃²x4th, ∂̃²y4th, ∂̃²z4th
    export ∇4th, div4th, ∇²4th
    export ∇̃4th, diṽ4th, ∇̃²4th
end

include("utils/abstract_types.jl")


# Traits
include("traits/boundarycondition.jl")
include("traits/shooting.jl")
include("traits/snappable.jl")
include("traits/grid.jl")

# Utils
include("utils/interpolations.jl")
include("utils/grids.jl")
include("utils/utils.jl")
include("utils/checks.jl")
include("utils/fields.jl")
include("utils/checkpointers.jl")
include("utils/snapshotter.jl")
include("utils/printinfo.jl")
include("utils/mute_grad.jl")

# Shots
include("shots/sources.jl")
include("shots/receivers.jl")
include("shots/shot.jl")

# General models
include("models/bdc_params.jl")
include("models/cpmlcoeffs.jl")
include("models/genparameters.jl")

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
include("models/elastic/ela_gradient.jl")
include("models/elastic/ela_init_bc.jl")

# Backend selection
include("models/backend_selection.jl")

# Inversion 
include("inversion/misfits/L2Misfit.jl")
include("inversion/misfits/CCTSMisfit.jl")

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
include("models/elastic/backends/Elastic2D_Iso_CPML_Threads.jl")

## HMC stuff
include("HMCseiswaves.jl")

using .HMCseiswaves
export AcouWavCDProb
export FiniteDifferencesMacros
export FDGeneratedFunctions

end # module
