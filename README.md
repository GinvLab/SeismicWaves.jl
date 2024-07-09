# SeismicWaves

[![Docs Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliageoph.gitlab.io/SeismicWaves.jl/stable)
[![Docs Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://juliageoph.gitlab.io/SeismicWaves.jl/dev)

SeismicWaves.jl is a Julia package for acoustic and elastic wave propagation simulations designed to be used in a Full-Waveform Inversion framework.

The main features of SeismicWaves.jl are:
- forward wave simulations with multiple sources and receivers (shots)
- 1D, 2D, and 3D acoustic wave simulations (constant density and variable density)
- 2D P-SV elastic wave similations (isotropic)
- gradients of misfit functions with respect to model parameters using the adjoint method (only acoustic simulations supported as of now)
- checkpointing of forward simulations for adjoint method
- device agnostic backends (CPUs or GPUs) thanks to [`ParallelStencil.jl`](https://github.com/omlins/ParallelStencil.jl)

More information and an extensive list of features can be found in the documentation, which you can either find [online](https://juliageoph.gitlab.io/SeismicWaves.jl/stable) or build locally by running the docs/make.jl file.

Warning: **Documentation is currently minimal**!
