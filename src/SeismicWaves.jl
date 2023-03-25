"""
Module for generalized wave propagation solvers.
"""
module SeismicWaves

using LinearAlgebra
using Printf
using ParallelStencil

#export Acoustic_CD_CPML_WaveSimul1D, Acoustic_CD_CPML_WaveSimul2D, Acoustic_CD_CPML_WaveSimul3D
export InputParametersAcoustic
#export InputBCParametersAcousticReflective,
export CPML_BC, Refl_BC
export Sources, Receivers
export swforward!, swmisfit!, swgradient!
export gaussource1D, rickersource1D


include("abstract_types.jl")
include("sources.jl")
include("receivers.jl")
include("check.jl")
include("precompute.jl")
include("init_shot.jl")
include("solve.jl")


include("models/cpmlcoeffs.jl")

include("traits/equation.jl")
include("traits/boundarycondition.jl")
include("traits/shooting.jl")
include("traits/snappable.jl")
include("traits/grid.jl")

include("models/acoustic/acou_models.jl")
include("models/acoustic/acou_params.jl")
include("models/acoustic/acou_check.jl")
include("models/acoustic/acou_precompute.jl")
include("models/acoustic/acou_forward.jl")
include("models/acoustic/acou_gradient.jl")
include("models/acoustic/acou_init_bc.jl")

include("wrappers.jl")

include("models/acoustic/backends/Acoustic1D.jl")

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

include("utils.jl")

include("HMCseiswaves.jl")

end
