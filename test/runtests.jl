# NOTE: many parts of this file take inspiration from the same file in the packages MPI.jl

# the following lines allow for precompiling of depencencies, making tests run faster
push!(LOAD_PATH, "../src")
using SeismicWaves: SeismicWaves

using Test, TestSetExtensions

# Run all specified tests
@testset ExtendedTestSet "SeismicWaves Tests" begin
    @includetests ARGS
end
