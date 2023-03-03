function build_model(params::InputParameters, vel::AbstractArray)
    # Select equation type (only acoustic supported as of now)
    if params isa InputParametersAcoustic
        # Select model dimensionality
        if params isa InputParametersAcoustic1D
            # Select boundary condition
            if params.boundcond == "reflective"
                model = IsotropicAcousticReflectiveWaveModel1D(
                    params.ntimesteps,
                    params.dt,
                    params.dh,
                    vel;
                    snapevery=params.savesnapshot ? params.snapevery : nothing,
                    infoevery=params.infoevery
                )
            elseif params.boundcond == "CPML"
                model = IsotropicAcousticCPMLWaveModel1D(
                    params.ntimesteps,
                    params.dt,
                    params.dh,
                    20,         # defaul parameters (TODO make it a choice in input parameters)
                    0.0001,
                    vel;
                    snapevery=params.savesnapshot ? params.snapevery : nothing,
                    infoevery=params.infoevery
                )
            else
                error("Not implemented boundary condition for this parameters")
            end
        elseif params isa InputParametersAcoustic2D
            # Select boundary condition
            if params.boundcond == "CPML"
                model = IsotropicAcousticCPMLWaveModel2D(
                    params.ntimesteps,
                    params.dt,
                    params.dh,
                    params.dh,
                    20,
                    0.0001,
                    vel;
                    freetop=params.freeboundtop,
                    snapevery=params.savesnapshot ? params.snapevery : nothing,
                    infoevery=params.infoevery
                )
            else
                error("Not implemented boundary condition for this parameters")
            end
        elseif params isa InputParametersAcoustic3D
            # Select boundary condition
            if params.boundcond == "CPML"
                model = IsotropicAcousticCPMLWaveModel3D(
                    params.ntimesteps,
                    params.dt,
                    params.dh,
                    params.dh,
                    params.dh,
                    20,
                    0.0001,
                    vel;
                    freetop=params.freeboundtop,
                    snapevery=params.savesnapshot ? params.snapevery : nothing,
                    infoevery=params.infoevery
                )
            else
                error("Not implemented boundary condition for this parameters")
            end
        end
    end
    return model
end

function select_backend(model::WaveModel, use_GPU::Bool)
    if !use_GPU
        if WaveEquationTrait(model) isa IsotropicAcousticWaveEquation
            if model isa WaveModel1D
                backend = Acoustic1D_Threads
            elseif model isa WaveModel2D
                backend = Acoustic2D_Threads
            elseif model isa WaveModel3D
                backend = Acoustic3D_Threads
            else
                error("Backend not found!")
            end
        end
    else
        if WaveEquationTrait(model) isa IsotropicAcousticWaveEquation
            if model isa WaveModel1D
                backend = Acoustic1D_CUDA
            elseif model isa WaveModel2D
                backend = Acoustic2D_CUDA
            elseif model isa WaveModel3D
                backend = Acoustic3D_CUDA
            else
                error("Backend not found!")
            end
        end
    end
    return backend
end

function forward!(
    params::InputParameters,
    vel::AbstractArray,
    shots::Vector{<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}};
    use_GPU = false
    )::Union{Vector{Array}, Nothing}
    # Build model
    model = build_model(params, vel)
    # Select backend
    backend = select_backend(model, use_GPU)
    # Solve simulation
    forward!(model, shots, backend)
end

function misfit!(
    params::InputParameters,
    vel::AbstractArray,
    shots::Vector{<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}};
    use_GPU = false
    )::Real
    # Build model
    model = build_model(params, vel)
    # Select backend
    backend = select_backend(model, use_GPU)
    @show backend
    # Compute misfit
    misfit!(model, shots, backend)
end

function gradient!(
    params::InputParameters,
    vel::AbstractArray,
    shots::Vector{<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}};
    use_GPU = false,
    check_freq::Union{Integer, Nothing} = nothing
    )::AbstractArray
    # Build model
    model = build_model(params, vel)
    # Select backend
    backend = select_backend(model, use_GPU)
    # Solve simulation
    gradient!(model, shots, backend; check_freq=check_freq)
end