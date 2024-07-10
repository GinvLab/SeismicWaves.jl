# NOTE: many parts of this file take inspiration from the same file in the packages MPI.jl

# the following lines allow for precompiling of depencencies, making tests run faster
push!(LOAD_PATH, "../src")

using Test, TestSetExtensions
using DSP, NumericalIntegration, LinearAlgebra
using Logging
using SeismicWaves

# Load CUDA only if requested for testing
if "CUDA" in ARGS
    using CUDA
end

include("utils/setup_models.jl")

# Run all tests
@testset ExtendedTestSet "SeismicWaves Tests" begin
    @includetests
end
