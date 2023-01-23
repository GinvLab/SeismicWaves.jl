# NOTE: many parts of this file take inspiration from the same file in the packages MPI.jl

# the following lines allow for precompiling of depencencies, making tests run faster
push!(LOAD_PATH, "../src")
import Waves

# list of files to NOT be tested
excludedfiles = []

function runtests()
    exename   = joinpath(Sys.BINDIR, Base.julia_exename())
    testdir   = pwd()
    # getting all test files to run tests on
    istest(f) = endswith(f, ".jl") && startswith(basename(f), "test_")
    testfiles = sort(filter(istest, vcat([joinpath.(root, files) for (root, dirs, files) in walkdir(testdir)]...)))

    printstyled("Testing package Waves.jl\n"; bold=true, color=:white)

    nfails = 0
    for f in testfiles
        println("")
        # skip excluded files
        if f ∈ excludedfiles
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