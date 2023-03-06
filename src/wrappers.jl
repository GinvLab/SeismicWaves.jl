function build_model(params::InputParametersAcoustic1D, vel::AbstractArray)
    if params.boundcond == "reflective"
        return IsotropicAcousticReflectiveWaveModel1D(
            params.ntimesteps,
            params.dt,
            params.dh,
            vel;
            snapevery=params.savesnapshot ? params.snapevery : nothing,
            infoevery=params.infoevery
        )
    elseif params.boundcond == "CPML"
        return IsotropicAcousticCPMLWaveModel1D(
            params.ntimesteps,
            params.dt,
            params.dh,
            20,         # default parameters (TODO make it a choice in input parameters)
            0.0001,
            vel;
            snapevery=params.savesnapshot ? params.snapevery : nothing,
            infoevery=params.infoevery
        )
    end
    error("Not implemented boundary condition for this parameters")
end

function build_model(params::InputParametersAcoustic2D, vel::AbstractArray)
    if params.boundcond == "CPML"
        return IsotropicAcousticCPMLWaveModel2D(
            params.ntimesteps,
            params.dt,
            params.dh,
            params.dh,
            20,         # default parameters (TODO make it a choice in input parameters)
            0.0001,
            vel;
            freetop=params.freeboundtop,
            snapevery=params.savesnapshot ? params.snapevery : nothing,
            infoevery=params.infoevery
        )
    end
    error("Not implemented boundary condition for this parameters")
end

function build_model(params::InputParametersAcoustic3D, vel::AbstractArray)
    if params.boundcond == "CPML"
        return IsotropicAcousticCPMLWaveModel3D(
            params.ntimesteps,
            params.dt,
            params.dh,
            params.dh,
            params.dh,
            20,         # default parameters (TODO make it a choice in input parameters)
            0.0001,
            vel;
            freetop=params.freeboundtop,
            snapevery=params.savesnapshot ? params.snapevery : nothing,
            infoevery=params.infoevery
        )
    end
    error("Not implemented boundary condition for this parameters")
end

select_backend(_::IsotropicAcousticWaveEquation, _::WaveModel1D, use_GPU::Bool) = (use_GPU ? Acoustic1D_CUDA : Acoustic1D_Threads)
select_backend(_::IsotropicAcousticWaveEquation, _::WaveModel2D, use_GPU::Bool) = (use_GPU ? Acoustic2D_CUDA : Acoustic2D_Threads)
select_backend(_::IsotropicAcousticWaveEquation, _::WaveModel3D, use_GPU::Bool) = (use_GPU ? Acoustic3D_CUDA : Acoustic3D_Threads)
select_backend(model::WaveModel, use_GPU::Bool) = select_backend(WaveEquationTrait(model), model, use_GPU)

@doc raw"""
    forward!(
        params::InputParameters,
        vel::AbstractArray,
        shots::Vector{<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}};
        use_GPU::Bool = false
    )::Union{Vector{AbstractArray}, Nothing}

Compute forward simulation using the given input parameters `params` and velocity model `vel` on multiple shots.

The flag `use_GPU` controls which backend is used for computation: the `CUDA.jl` GPU backend if `true`, otherwise the standard `Base.Threads` CPU backend.

Receivers traces are stored in the `Receivers` object for each shot. See also [`Receivers`](@ref).

Return a vector of snapshots for every shot if snapshotting is enabled.

See also [`Sources`](@ref), [`Receivers`](@ref).
"""
function forward!(
    params::InputParameters,
    vel::AbstractArray,
    shots::Vector{<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}};
    use_GPU::Bool = false
    )::Union{Vector{AbstractArray}, Nothing}
    # Build model
    model = build_model(params, vel)
    # Select backend
    backend = select_backend(model, use_GPU)
    # Solve simulation
    forward!(model, shots, backend)
end

@doc raw"""
    misfit!(
        params::InputParameters,
        vel::AbstractArray,
        shots::Vector{<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}};
        use_GPU::Bool = false
    )::Real

Return the misfit w.r.t. observed data by running a forward simulation using the given input parameters `params` and velocity model `vel` on multiple shots.

The flag `use_GPU` controls which backend is used for computation: the `CUDA.jl` GPU backend if `true`, otherwise the standard `Base.Threads` CPU backend.

Receivers traces are stored in the `Receivers` object for each shot.
    
See also [`Sources`](@ref), [`Receivers`](@ref), [`forward!`](@ref).
"""
function misfit!(
    params::InputParameters,
    vel::AbstractArray,
    shots::Vector{<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}};
    use_GPU::Bool = false
    )::Real
    # Build model
    model = build_model(params, vel)
    # Select backend
    backend = select_backend(model, use_GPU)
    @show backend
    # Compute misfit
    misfit!(model, shots, backend)
end

@doc raw"""
    gradient!(
        params::InputParameters,
        vel::AbstractArray,
        shots::Vector{<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}};
        use_GPU::Bool = false,
        check_freq::Union{Integer, Nothing} = nothing
    )::AbstractArray

Compute gradients w.r.t. model parameters using the given input parameters `params` and velocity model `vel` on multiple shots.

The flag `use_GPU` controls which backend is used for computation: the `CUDA.jl` GPU backend if `true`, otherwise the standard `Base.Threads` CPU backend.

The `check_freq` parameter controls the checkpoiting frequency for adjoint computation.
If `nothing`, no checkpointing is performed.
If greater than 2, a checkpoint is saved every `check_freq` time step.
The optimal tradeoff value is `check_freq = sqrt(nt)` where `nt` is the number of time steps of the forward simulation.
Bigger values speed up computation at the cost of using more memory.

See also [`Sources`](@ref), [`Receivers`](@ref), [`forward!`](@ref), [`misfit!`](@ref).
"""
function gradient!(
    params::InputParameters,
    vel::AbstractArray,
    shots::Vector{<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}};
    use_GPU::Bool = false,
    check_freq::Union{Integer, Nothing} = nothing
    )::AbstractArray
    # Build model
    model = build_model(params, vel)
    # Select backend
    backend = select_backend(model, use_GPU)
    # Solve simulation
    gradient!(model, shots, backend; check_freq=check_freq)
end