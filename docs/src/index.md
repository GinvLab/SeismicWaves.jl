
```@meta
Author = "Andrea Zunino"
Author = "Giacomo Aloisi"
EditURL = "https://gitlab.com/JuliaGeoph/SeismicWaves.jl/-/tree/main/docs/src/"
```

# Contents

```@contents
Pages = ["index.md"]
Depth = 4
```

# User guide

**SeismicWaves** is a Julia package for seismic wave propagation using the finite difference method.

This package provides some functions to solve inverse problems using the Hamiltonian Monte Carlo (HMC) method, as part of the [`HMCLab`](https://gitlab.com/JuliaGeoph/HMCLab.jl) project (see the  [`MCsamplers.jl`](https://gitlab.com/JuliaGeoph/MCsamplers.jl) package). 



## Public API

```@docs
SeismicWaves
InputParametersAcoustic
InputParametersAcousticVariableDensity
CPMLBoundaryConditionParameters
ReflectiveBoundaryConditionParameters
VpAcousticCDMaterialProperty
VpRhoAcousticVDMaterialProperty
WaveSimul
ScalarSources
ScalarReceivers
Shot
swforward!
swmisfit!
swgradient!
build_wavesim
gaussource1D 
gaussdersource1D
rickersource1D
```

