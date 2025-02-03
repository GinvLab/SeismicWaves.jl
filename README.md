# SeismicWaves

[![Docs Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ginvlab.github.io/SeismicWaves.jl/stable)
[![Docs Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://ginvlab.github.io/SeismicWaves.jl/dev)

SeismicWaves.jl is a Julia package for acoustic and elastic wave propagation simulations designed to be used in a Full-Waveform Inversion framework. It solves different flavours of the wave equation using the finite difference method.

The main features of SeismicWaves.jl are:
- checkpointing of forward simulations for adjoint method
- device agnostic backends (CPUs or GPUs) thanks to [`ParallelStencil.jl`](https://github.com/omlins/ParallelStencil.jl)

This package additionally provides some functions to solve inverse problems using the Hamiltonian Monte Carlo (HMC) method, as part of the [`G⁻¹Lab`](https://ginvlab.github.io) framework. 

More information and an extensive list of features can be found in the documentation, which you can either find [online](https://ginvlab.github.io/SeismicWaves.jl/stable) or build locally by running the docs/make.jl file.

Warning: **Documentation is currently minimal** and work in progress!
