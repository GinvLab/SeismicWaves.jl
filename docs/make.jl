
using Documenter, SeismicWaves


makedocs(repo=Remotes.GitLab("JuliaGeoph","SeismicWaves.jl"), # "https://gitlab.com/JuliaGeoph/SeismicWaves.jl/blob/{commit}{path}#{line}",
         sitename="SeismicWaves.jl",
         modules = [SeismicWaves],
         authors = "Andrea Zunino, Giacomo Aloisi",
         format = Documenter.HTML(prettyurls=get(ENV,"CI",nothing)=="true"),
         pages = [
             "Home" => "index.md",
         ],
         warnonly = [:missing_docs, :cross_references]
         )

deploydocs(
    repo="gitlab.com/JuliaGeoph/SeismicWaves.jl.git",
    devbranch = "main",
    deploy_config = Documenter.GitLab(),
    branch = "gl-pages"
)
