
#######################################################


@doc raw"""
    swforward!(
        params::InputParameters,
        vel::AbstractArray,
        shots::Vector{<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}};
        parall::Symbol= :threads,
        snapevery::Union{Int, Nothing} = nothing,
        infoevery::Union{Int, Nothing} = nothing
    )::Union{Vector{AbstractArray}, Nothing}

Compute forward simulation using the given input parameters `params` and velocity model `vel` on multiple shots.

The `parall::Symbol` controls which backend is used for computation:
  - the `CUDA.jl` GPU backend if set to `GPU`
  - `Base.Threads` CPU threads if set to `:threads`
  - otherwise the serial version if set to `:serial`

Receivers traces are stored in the `Receivers` object for each shot. See also [`Receivers`](@ref).

Return a vector of snapshots for every shot if snapshotting is enabled.

See also [`Sources`](@ref), [`Receivers`](@ref).

# Keyword arguments
- `use_GPU::Bool = false`: controls which backend is used (`true` for GPU backend, `false` for CPU backend).
- `snapevery::Union{Int, Nothing} = nothing`: if specified, saves itermediate snapshots at the specified frequency (one every `snapevery` time step iteration) and return them as a vector of arrays  
- `infoevery::Union{Int, Nothing} = nothing`: if specified, logs info about the current state of simulation every `infoevery` time steps.
"""
function swforward!(
    params::InputParameters,
    vel::AbstractArray,
    shots::Vector{<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}};
    parall::Symbol = :threads,
    snapevery::Union{Int, Nothing} = nothing,
    infoevery::Union{Int, Nothing} = nothing
    )::Union{Vector{AbstractArray}, Nothing}
    # Build wavesim
    wavesim = build_wavesim(params, vel; snapevery=snapevery, infoevery=infoevery)
    # Select backend
    backend = select_backend(wavesim, parall)
    # Solve simulation
    run_swforward!(wavesim, backend, shots)
end

#######################################################

@doc raw"""
    swmisfit!(
        params::InputParameters,
        vel::AbstractArray,
        shots::Vector{<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}};
        parall::Symbol= :threads,
    )::Real

Return the misfit w.r.t. observed data by running a forward simulation using the given input parameters `params` and velocity model `vel` on multiple shots.

The `parall::Symbol` controls which backend is used for computation:
  - the `CUDA.jl` GPU backend if set to `GPU`
  - `Base.Threads` CPU threads if set to `:threads`
  - otherwise the serial version if set to `:serial`

Receivers traces are stored in the `Receivers` object for each shot.
    
See also [`Sources`](@ref), [`Receivers`](@ref), [`swforward!`](@ref).
"""
function swmisfit!(
    params::InputParameters,
    vel::AbstractArray,
    shots::Vector{<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}};
    parall::Symbol = :threads,
    )::Real
    # Build wavesim
    wavesim = build_wavesim(params, vel)
    # Select backend
    backend = select_backend(wavesim, parall)
    # Compute misfit
    run_swmisfit!(wavesim, backend, shots)
end

#######################################################

@doc raw"""
    swgradient!(
        params::InputParameters,
        vel::AbstractArray,
        shots::Vector{<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}};
        parall::Symbol = :threads,
        check_freq::Union{Int, Nothing} = nothing,
        infoevery::Union{Int, Nothing} = nothing
    )::AbstractArray

Compute gradients w.r.t. model parameters using the given input parameters `params` and velocity model `vel` on multiple shots.

The flag `use_GPU` controls which backend is used for computation: the `CUDA.jl` GPU backend if `true`, otherwise the standard `Base.Threads` CPU backend.

The `check_freq` parameter controls the checkpoiting frequency for adjoint computation.
If `nothing`, no checkpointing is performed.
If greater than 2, a checkpoint is saved every `check_freq` time step.
The optimal tradeoff value is `check_freq = sqrt(nt)` where `nt` is the number of time steps of the forward simulation.
Bigger values speed up computation at the cost of using more memory.

See also [`Sources`](@ref), [`Receivers`](@ref), [`swforward!`](@ref), [`swmisfit!`](@ref).

# Keyword arguments
- `use_GPU::Bool = false`: controls which backend is used (`true` for GPU backend, `false` for CPU backend).
- `check_freq::Union{Int, Nothing}`: if specified, enables checkpointing and specifies the checkpointing frequency.
- `infoevery::Union{Int, Nothing} = nothing`: if specified, logs info about the current state of simulation every `infoevery` time steps.
"""
function swgradient!(
    params::InputParameters,
    vel::AbstractArray,
    shots::Vector{<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}};
    parall::Symbol = :threads,
    check_freq::Union{Int, Nothing} = nothing,
    infoevery::Union{Int, Nothing} = nothing
    )::AbstractArray
                  # Build wavesim
                  wavesim = build_wavesim(params, vel; infoevery=infoevery)
    # Select backend
    backend = select_backend(wavesim, parall)
    # Solve simulation
    run_swgradient!(wavesim, backend, shots; check_freq=check_freq)
end


#######################################################


build_wavesim(params::InputParametersAcoustic, vel::AbstractArray; kwargs...) = build_wavesim(params, params.boundcond, vel; kwargs...)


function build_wavesim(params::InputParametersAcoustic, cpmlparams::CPML_BC, vel::AbstractArray; kwargs...)

    N = length(params.gridsize)

    acoumod = Acoustic_CD_CPML_WaveSimul{N}(
        params.ntimesteps,
        params.dt,
        params.gridspacing,
        cpmlparams.halo,
        cpmlparams.rcoef,
        vel;
        freetop=cpmlparams.freeboundtop,
        kwargs...
            )
    return acoumod
end

#######################################################

function select_backend(wavesim,parall)

    parasym = [:serial, :threads, :GPU]
    if !(parall in parasym)
        throw(ErrorException("Argument `parall` must be one of the following symbols: $parasym. Got $(parall)."))
    end

    tpwavsim = typeof(wavesim)

    physsim = [Acoustic_CD_CPML_WaveSimul{1}, Acoustic_CD_CPML_WaveSimul{2}, Acoustic_CD_CPML_WaveSimul{3} ]
    if !(tpwavsim in physsim)
        throw(ErrorException("$(typeof(wavesim)) not (yet?) implemented."))
    end

    if tpwavsim <: Acoustic_CD_WaveSimul

        if tpwavsim==Acoustic_CD_CPML_WaveSimul{1}
            if parall==:serial
                return Acoustic1D_CD_CPML_Serial
            elseif parall==:threads
                return Acoustic1D_CD_CPML_Threads
            elseif parall==:GPU
                return Acoustic1D_CD_CPML_GPU
            end                

        elseif tpwavsim==Acoustic_CD_CPML_WaveSimul{2}
            if parall==:serial
                return Acoustic2D_CD_CPML_Serial
            elseif parall==:threads
                return Acoustic2D_CD_CPML_Threads
            elseif parall==:GPU
                return Acoustic2D_CD_CPML_GPU
            end     

        elseif tpwavsim==Acoustic_CD_CPML_WaveSimul{3}
            if parall==:serial
                # return Acoustic3D_CD_CPML_Serial
                error("Acoustic3D_CD_CPML_Serial not yet implemented...")
                return nothing
            elseif parall==:threads
                return Acoustic3D_CD_CPML_Threads
            elseif parall==:GPU
                return Acoustic3D_CD_CPML_GPU
            end   

        end

    end
    return
end

#######################################################

# build_wavesim(params::InputParametersAcoustic{1}, cpmlparams::CPML_BC, vel::AbstractArray; kwargs...) =
#     Acoustic_CD_CPML_WaveSimul1D(
#         params.ntimesteps,
#         params.dt,
#         params.gridspacing,
#         cpmlparams.halo,
#         cpmlparams.rcoef,
#         vel;
#         kwargs...
#     )
# build_wavesim(params::InputParametersAcoustic{2}, cpmlparams::CPML_BC, vel::AbstractArray; kwargs...) =
#     Acoustic_CD_CPML_WaveSimul2D(
#         params.ntimesteps,
#         params.dt,
#         params.gridspacing,
#         cpmlparams.halo,
#         cpmlparams.rcoef,
#         vel;
#         freetop=cpmlparams.freeboundtop,
#         kwargs...
#     )
# build_wavesim(params::InputParametersAcoustic{3}, cpmlparams::CPML_BC, vel::AbstractArray; kwargs...) =
#     Acoustic_CD_CPML_WaveSimul3D(
#         params.ntimesteps,
#         params.dt,
#         params.gridspacing,
#         cpmlparams.halo,
#         cpmlparams.rcoef,
#         vel;
#         freetop=cpmlparams.freeboundtop,
#         kwargs...
#     )


# select_backend(wavesim::WaveSimul, parall::Symbol) = select_backend(WaveEquationTrait(wavesim), wavesim, parall)

# function select_backend(physics, wavesim, parall)

#     if !(parall in [:serial, :threads, :GPU])
#         throw(ErrorException("Argument `parall` must be one of the following symbols: :serial, :threads, :GPU"))
#     end

#     if !(wavesim in [:Acoustic_CD_WaveSimul, :ElasticWaveEquation])
#         throw(ErrorException("Argument `wavesim` must be one of the following: WaveSimul1D, WaveSimul1D or WaveSimul1D"))
#     end

#     if typeof(physics)==Acoustic_CD_WaveSimul

#         if wavesim==WaveSimul1D
#             if parall==:serial
#                 return Acoustic1D_serial
#             elseif parall==:threads
#                 return Acoustic1D_CD_CPML_Threads
#             elseif parall==:GPU
#                 return Acoustic1D_CD_CPML_GPU
#             end                

#         elseif wavesim==WaveSimul2D
#             if parall==:serial
#                 return Acoustic2D_serial
#             elseif parall==:threads
#                 return Acoustic2D_CD_CPML_Threads
#             elseif parall==:GPU
#                 return Acoustic2D_CD_CPML_GPU
#             end     

#         elseif wavesim==WaveSimul3D
#             if parall==:serial
#                 return Acoustic3D_serial
#             elseif parall==:threads
#                 return Acoustic3D_CD_CPML_Threads
#             elseif parall==:GPU
#                 return Acoustic3D_CD_CPML_GPU
#             end   

#         end

#     elseif typeof(physics)==ElasticWaveEquation
#         @error ("Elastic wave propagation is still work in progress...")
#         return

#     end
# end

# select_backend(_::Acoustic_CD_WaveSimul, _::WaveSimul1D, parall::Symbol) = (use_GPU ? Acoustic1D_CD_CPML_GPU : Acoustic1D_CD_CPML_Threads)
# select_backend(_::Acoustic_CD_WaveSimul, _::WaveSimul2D, parall::Symbol) = (use_GPU ? Acoustic2D_CD_CPML_GPU : Acoustic2D_CD_CPML_Threads)
# select_backend(_::Acoustic_CD_WaveSimul, _::WaveSimul3D, parall::Symbol) = (use_GPU ? Acoustic3D_CD_CPML_GPU : Acoustic3D_CD_CPML_Threads)
