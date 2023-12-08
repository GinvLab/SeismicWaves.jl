
```@meta
EditURL = "https://gitlab.com/JuliaGeoph/SeismicWaves.jl/-/tree/main/docs/src/"
```

# Contents

```@contents
Pages = ["index.md"]
Depth = 4
```

# User guide

**SeismicWaves** is a Julia package for seismic wave propagation using the finite difference method. `SeismicWaves` provides functions to solve the forward problem and the gradient of a misfit functional with respect to model parameters.
This package additionally provides some functions to solve inverse problems using the Hamiltonian Monte Carlo (HMC) method, as part of the [`HMCLab`](https://gitlab.com/JuliaGeoph/HMCLab.jl) framework. 



## Public API

```@autodocs
Modules = [SeismicWaves]
Private = false
```



