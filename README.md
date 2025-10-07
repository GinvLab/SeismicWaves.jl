# SeismicWaves

[![Docs Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ginvlab.github.io/SeismicWaves.jl/stable)
[![Docs Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://ginvlab.github.io/SeismicWaves.jl/dev)

SeismicWaves.jl is a Julia package for acoustic and elastic wave propagation simulations designed to be used in a full-waveform inversion (FWI) framework. It solves different flavours of the wave equation using the finite difference method.

![Wave propagation example](../docs/src/images/acoustic2d_xpu_complex_freetop_halo20_360.gif)

The main features of SeismicWaves.jl are:
- acoustic and elastic simulations
- gradients using the adjoint method
- different misfit/objective functionals provided or the possibility to specifing your own
- device-agnostic backends for parallelization, i.e., CPUs or GPUs (CUDA, AMDGPU, Metal - Apple M-chips) thanks to [`ParallelStencil.jl`](https://github.com/omlins/ParallelStencil.jl) 
- C-PML boundary conditions
- checkpointing of simulations for the adjoint method

Work in progress:
- noise correlations (seismic interferometry)
- arbitrary-order stencils
- 3D elastic simulations

This package integrates well with the inversion framework provided by [`InverseAlgos`](https://github.com/GinvLab/InverseAlgos.jl), which includes both deterministic (L-BFGS, Quasi-Newton, etc.) and probabilistic (e.g., Hamiltonian Monte Carlo) inverse algorithms, as part of the [`G⁻¹Lab`](https://ginvlab.github.io) framework. 

More information and an extensive list of features can be found in the documentation, which can be found either [online](https://ginvlab.github.io/SeismicWaves.jl/stable) or built locally by running the docs/make.jl file.

> [!WARNING]
> **Documentation is currently minimal** and work in progress!
