# NOTE: many parts of this file take inspiration from the same file in the packages MPI.jl

# the following lines allow for precompiling of depencencies, making tests run faster
push!(LOAD_PATH, "../src")
import SeismicWaves

import CUDA

# list of files to NOT be tested
excludedfiles = []
# list of files to skipped ONLY if CUDA is NOT functional
if !CUDA.functional()
    push!(excludedfiles, joinpath(pwd(), "test_analytical_CUDA.jl"))
    push!(excludedfiles, joinpath(pwd(), "test_gradient_CUDA.jl"))
end

function runtests()
    exename   = joinpath(Sys.BINDIR, Base.julia_exename())
    testdir   = pwd()
    # getting all test files to run tests on
    istest(f) = endswith(f, ".jl") && startswith(basename(f), "test_")
    testfiles = sort(filter(istest, vcat([joinpath.(root, files) for (root, dirs, files) in walkdir(testdir)]...)))

    printstyled("Testing package SeismicWaves.jl\n"; bold=true, color=:white)

    nfails = 0
    for f in testfiles
        println("")
        # skip excluded files
        if f in excludedfiles
            printstyled("Skipping $(basename(f))\n"; bold=true, color=:white)
            continue
        end
        # run tests on file
        try
            printstyled("Testing $(basename(f))\n"; bold=true, color=:white)
            run(`$exename -O3 --startup-file=no --check-bounds=yes $f`)
        catch ex
            nfails += 1
        end
    end

    return nfails
end

exit(runtests())