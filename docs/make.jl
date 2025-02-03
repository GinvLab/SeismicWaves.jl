
using SeismicWaves

import Literate
using Documenter

println("Converting examples...")

Literate.markdown(
    joinpath(@__DIR__, "src", "examples.jl"), joinpath(@__DIR__, "src");
    credit = false
)

println("Building documentation...")

makedocs(; repo=Remotes.GitHub("GinvLab", "SeismicWaves.jl"), 
    sitename="SeismicWaves.jl",
    modules=[SeismicWaves],
    authors="Giacomo Aloisi, Andrea Zunino",
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true"),
    pages=[
        "Home" => "index.md",
        "User guide" => "guide.md",
        "Examples" => "examples.md",
        "API" => "api.md"
    ],
    warnonly=[:missing_docs, :cross_references]
)

deploydocs(;
    repo="github.com/GinvLab/SeismicWaves.jl.git",
    devbranch="main",
    deploy_config=Documenter.GitHubActions(),
    branch="gh-pages"
)
