@doc """

$(TYPEDSIGNATURES)

Return the misfit w.r.t. observed data by running a forward simulation using the given input parameters `params` and material properties `matprop` on multiple shots.
Receivers traces are stored in the receivers for each shot.

# Positional arguments
- `params::InputParameters{T, N}`: input parameters for the simulation, where T represents the data type and N represents the number of dimensions. They vary depending on the simulation kind (e.g., acoustic variable-density).
- `matprop::MaterialProperties{T, N}`: material properties for the simulation, where T represents the data type and N represents the number of dimensions. They vary depending on the simulation kind (e.g., Vp only is required for an acoustic constant-density simulation).
- `shots::Vector{<:Shot{T}}`: a vector whose elements are `Shot` structures. Each shot contains information about both source(s) and receiver(s).

# Keyword arguments
- `parall::Symbol = :threads`: controls which backend is used for computation:
    - the `CUDA.jl` GPU backend performing automatic domain decomposition if set to `:CUDA`
    - the `AMDGPU.jl` GPU backend performing automatic domain decomposition if set to `:AMDGPU`
    - the `Metal.jl` GPU backend performing automatic domain decomposition if set to `:Metal`
    - `Base.Threads` CPU threads performing automatic domain decomposition if set to `:threads`
    - `Base.Threads` CPU threads sending a group of sources to each thread if set to `:threadpersrc`
    - otherwise the serial version if set to `:serial`
- `logger::Union{Nothing,AbstractLogger}`: specifies the logger to be used.

See also [`InputParameters`](@ref), [`MaterialProperties`](@ref) and [`Shot`](@ref).
See also [`swforward!`](@ref) and [`swgradient!`](@ref) and [`Shot`](@ref).
"""
function swmisfit!(
    params::InputParameters{T, N},
    matprop::MaterialProperties{T, N},
    shots::Vector{<:Shot{T}};
    parall::Symbol=:threads,
    misfit::AbstractMisfit=L2Misfit(nothing),
    logger::Union{Nothing, AbstractLogger}=nothing
)::T where {T, N}
    if logger === nothing
        logger = current_logger()
    end
    return with_logger(logger) do
        # Build wavesim
        wavesim = build_wavesim(params, matprop; parall=parall, gradient=false)
        # Compute misfit
        run_swmisfit!(wavesim, matprop, shots; misfit=misfit)
    end
end

@doc """

$(TYPEDSIGNATURES)

Return the misfit w.r.t. observed data by running a forward simulation using the given `WaveSimulation` object as an input.
Receivers traces are stored in the receivers for each shot. See also [`build_wavesim`](@ref) on how to build the `WaveSimulation`.

# Positional arguments
- `wavesim::Union{WaveSimulation{T,N},Vector{<:WaveSimulation{T,N}}}`: input `WaveSimulation` object containing all required information to run the simulation.
- `matprop::MaterialProperties{T, N}`: material properties for the simulation, where T represents the data type and N represents the number of dimensions. They vary depending on the simulation kind (e.g., Vp only is required for an acoustic constant-density simulation).
- `shots::Vector{<:Shot{T}}`: a vector whose elements are `Shot` structures. Each shot contains information about both source(s) and receiver(s).

# Keyword arguments
- `parall::Symbol = :threads`: controls which backend is used for computation:
    - the `CUDA.jl` GPU backend performing automatic domain decomposition if set to `:CUDA`
    - the `AMDGPU.jl` GPU backend performing automatic domain decomposition if set to `:AMDGPU`
    - the `Metal.jl` GPU backend performing automatic domain decomposition if set to `:Metal`
    - `Base.Threads` CPU threads performing automatic domain decomposition if set to `:threads`
    - `Base.Threads` CPU threads sending a group of sources to each thread if set to `:threadpersrc`
    - otherwise the serial version if set to `:serial`

See also [`InputParameters`](@ref), [`MaterialProperties`](@ref) and [`Shot`](@ref).
See also [`swforward!`](@ref) and [`swgradient!`](@ref) and [`Shot`](@ref).
"""
function swmisfit!(wavesim::Union{WaveSimulation{T, N}, Vector{<:WaveSimulation{T, N}}}, matprop::MaterialProperties{T, N}, shots::Vector{<:Shot{T}};
    logger::Union{Nothing, AbstractLogger}=nothing, kwargs...)::T where {T, N}
    if logger === nothing
        logger = current_logger()
    end
    return with_logger(logger) do
        run_swmisfit!(wavesim, matprop, shots; kwargs...)
    end
end

#######################################################

### MISFITS ###

## single or multiple WaveSimulation objects
function run_swmisfit!(
    model::Union{WaveSimulation{T, N}, Vector{<:WaveSimulation{T, N}}},
    matprop::MaterialProperties{T, N},
    shots::Vector{<:Shot{T}};
    misfit::AbstractMisfit=L2Misfit(nothing)
)::T where {T, N}

    # Solve forward model for all shots
    run_swforward!(model, matprop, shots)
    # Compute total misfit for all shots
    @info "Computing misfit"
    totmisfitval = 0
    for singleshot in shots
        @debug "Checking invcov matrix"
        if typeof(model) <: Vector{<:WaveSimulation}
            for i in eachindex(model)
                check_invcov_matrix(model[i], singleshot.recs.invcov)
            end
        else
            check_invcov_matrix(model, singleshot.recs.invcov)
        end
        totmisfitval += misfit(singleshot.recs, matprop)
    end

    return totmisfitval
end