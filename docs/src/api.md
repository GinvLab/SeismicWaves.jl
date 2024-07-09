# [API](@id api)

## Input parameters
```@autodocs
Modules = [SeismicWaves]
Private = false
Filter = t -> (typeof(t) === DataType || typeof(t) === UnionAll) && (t <: InputParameters)
```

## Input BDCs parameters
```@autodocs
Modules = [SeismicWaves]
Private = false
Filter = t -> (typeof(t) === DataType || typeof(t) === UnionAll) && (t <: InputBoundaryConditionParameters)
```

## Material properties
```@autodocs
Modules = [SeismicWaves]
Private = false
Filter = t -> (typeof(t) === DataType || typeof(t) === UnionAll) && (t <: MaterialProperties)
```

## Shots, sources and receivers
```@autodocs
Modules = [SeismicWaves]
Private = false
Filter = t -> (typeof(t) === DataType || typeof(t) === UnionAll) && (t <: Shot || t <: Sources || t <: Receivers)
```

## Solver functions

```@docs
swforward!
swmisfit!
swgradient!
```

## Utilities

```@docs
build_wavesim
gaussstf
gaussderivstf
rickerstf
```

