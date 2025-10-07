
@doc """

$(TYPEDEF)

RunParameters is a struct containing various parameters related to forward (and gradient) simulations.

# Keyword arguments

- `parall::Symbol = :threads`: controls which backend is used for computation:
    - the `CUDA.jl` GPU backend performing automatic domain decomposition if set to `:CUDA`
    - the `AMDGPU.jl` GPU backend performing automatic domain decomposition if set to `:AMDGPU`
    - the `Metal.jl` GPU backend performing automatic domain decomposition if set to `:Metal`
    - `Base.Threads` CPU threads performing automatic domain decomposition if set to `:threads`
    - `Base.Threads` CPU threads sending a group of sources to each thread if set to `:threadpersrc`
    - otherwise the serial version if set to `:serial`
- `snapevery::Union{<:Int, Nothing} = nothing`: if specified, saves itermediate snapshots at the specified frequency (one every `snapevery` time step iteration) and return them as a vector of arrays (only for forward simulations).
- `infoevery::Union{<:Int, Nothing} = nothing`: if specified, logs info about the current state of simulation every `infoevery` time steps.
- `logger::AbstractLogger=current_logger()`: specifies the logger to be used.
- `erroronCFL::Bool=true`: throw an error if the CFL condition is not met.
- `minPPW::Int=10`: minimum number of points per wavelength (PPW).
- `erroronPPW::Bool=true`: throw an error if the minimum number of points per wavelength (PPW) is not achieved (otherwise warn about it).

"""
struct RunParameters
    parall::Symbol
    snapevery::Union{Int, Nothing}
    infoevery::Union{Int, Nothing}
    logger::AbstractLogger
    erroronCFL::Bool
    minPPW::Int
    erroronPPW::Bool

    function RunParameters(;
                           parall::Symbol=:threads,
                           snapevery::Union{Int, Nothing}=nothing,
                           infoevery::Union{Int, Nothing}=nothing,
                           logger::AbstractLogger=current_logger(),
                           erroronCFL::Bool=true,
                           minPPW::Int=10,
                           erroronPPW::Bool=true                           
                           )
        @assert minPPW>=0
        return new(parall,snapevery,infoevery,logger,erroronCFL,minPPW,erroronPPW)
    end
end


@doc """

$(TYPEDEF)

GradParameters is a struct containing various parameters related specifically to gradient simulations.

# Keyword arguments

- `mute_radius_src::Int=0`: grid points inside a ball with radius specified by the parameter (in grid points) will have their gradient smoothed by a factor inversely proportional to their distance from *source* positions.
- `mute_radius_rec::Int=0`: grid points inside a ball with radius specified by the parameter (in grid points) will have their gradient smoothed by a factor inversely proportional to their distance from *receiver* positions.
- `compute_misfit::Bool=false`: default false. If true, also computes and returns the misfit value.
- `check_freq::Int=1`: if specified, enables checkpointing and defines the checkpointing frequency. Deafults to no checkpointing (=1).

"""
struct GradParameters
    mute_radius_src::Int
    mute_radius_rec::Int
    compute_misfit::Bool
    check_freq::Int

    function GradParameters(;
                            mute_radius_src::Int=0,
                            mute_radius_rec::Int=0,
                            compute_misfit::Bool=false,
                            check_freq::Int=1
                            )
        @assert check_freq>0
        return new(mute_radius_src,mute_radius_rec,compute_misfit,check_freq)
    end
end


