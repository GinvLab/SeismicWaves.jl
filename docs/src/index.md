
```@meta
EditURL = "https://gitlab.com/JuliaGeoph/SeismicWaves.jl/-/tree/main/docs/src/"
```

# SeismicWaves.jl

SeismicWaves.jl is a Julia package for acoustic and elastic wave propagation simulations designed to be used in a Full-Waveform Inversion framework. It solves different flavours of the wave equation using the finite difference method.

The main features of SeismicWaves.jl are:
- forward wave simulations with multiple pairs of sources and receivers (shots)
- 1D, 2D, and 3D acoustic wave simulations (constant density and variable density)
- 2D P-SV elastic wave similations (isotropic)
- free boundary and C-PML absorbing boundary conditions supported
- gradients of misfit functions with respect to model parameters using the adjoint method (only acoustic simulations supported as of now)
- checkpointing of forward simulations for adjoint method
- device agnostic backends (CPUs or GPUs) thanks to [`ParallelStencil.jl`](https://github.com/omlins/ParallelStencil.jl)

This package additionally provides some functions to solve inverse problems using the Hamiltonian Monte Carlo (HMC) method, as part of the [`HMCLab`](https://gitlab.com/JuliaGeoph/HMCLab.jl) framework. 

More information and an extensive list of features can be found in the documentation, which you can either find [online](https://juliageoph.gitlab.io/SeismicWaves.jl/stable) or build locally by running the docs/make.jl file.

> Warning: **Documentation is currently minimal** and work in progress!

See the [Examples](@ref examples) for how to use this package!

See the [API](@ref api) for detailed information about the functionalities of this package!
