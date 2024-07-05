###########################################################

# Functions for all ElasticIsoWaveSimulation subtypes

@views function check_matprop(model::ElasticIsoWaveSimulation{T, N}, matprop::ElasticIsoMaterialProperties{T, N}) where {T, N}
    # Checks
    vp = sqrt.((matprop.λ .+ 2.0 * matprop.μ) ./ matprop.ρ)
    @assert ndims(vp) == N "Material property dimensionality must be the same as the wavesim!"
    @assert size(vp) == model.grid.size "Material property number of grid points must be the same as the wavesim! \n $(size(matprop.vp)), $(model.grid.size)"
    @assert all(matprop.λ .> 0) "Lamè coefficient λ must be positive!"
    @assert all(matprop.μ .> 0) "Lamè coefficient μ must be positive!"
    @assert all(matprop.ρ .> 0) "Density must be positive!"

    # Check courant condition
    vel_max = get_maximum_func(model)(vp)
    tmp = sqrt.(sum(1 ./ model.grid.spacing .^ 2))
    courant = vel_max * model.dt * tmp
    @info "Courant number: $(courant)"
    if courant > 1.0
        @warn "Courant condition not satisfied! [$(courant)]"
    end

    return
end

function check_numerics(
    model::ElasticIsoWaveSimulation{T},
    shot::MomentTensorShot{T};
    min_ppw::Int=10
) where {T}
    # Check points per wavelengh
    # min Vs
    vel_min = get_minimum_func(model)(sqrt.(model.matprop.μ ./ model.matprop.ρ))
    h_max = maximum(model.grid.spacing)
    fmax = shot.srcs.domfreq * 2.0
    ppw = vel_min / (fmax * h_max)
    @info "Points per wavelength: $(ppw)"

    dh0 = round((vel_min / (min_ppw * fmax)); digits=2)
    @assert ppw >= min_ppw "Not enough points per wavelength (assuming fmax = 2*domfreq)! \n [$(round(ppw,digits=1)) instead of >= $min_ppw]\n  Grid spacing should be <= $dh0"
    return
end

@views function update_matprop!(model::ElasticIsoWaveSimulation{T, 2}, matprop::ElasticIsoMaterialProperties{T, 2}) where {T}

    # Update material properties
    copyto!(model.matprop.λ, matprop.λ)
    copyto!(model.matprop.μ, matprop.μ)
    copyto!(model.matprop.ρ, matprop.ρ)

    # the following on device?
    precomp_elaprop!(model)
end

function precomp_elaprop!(model::ElasticIsoWaveSimulation{T, 2}; harmonicaver_μ=true) where {T}
    # Excract fields
    ρ, λ, μ = model.grid.fields["ρ"].value, model.grid.fields["λ"].value, model.grid.fields["μ"].value
    ρ_ihalf_jhalf = model.grid.fields["ρ_ihalf_jhalf"].value
    μ_ihalf = model.grid.fields["μ_ihalf"].value
    μ_jhalf = model.grid.fields["μ_jhalf"].value
    λ_ihalf = model.grid.fields["λ_ihalf"].value
    # Copy from internal matprop
    copyto!(ρ, model.matprop.ρ)
    copyto!(λ, model.matprop.λ)
    copyto!(μ, model.matprop.μ)
    #-------------------------------------------------------------
    # pre-interpolate properties at half distances between nodes
    #-------------------------------------------------------------
    # ρ_ihalf_jhalf (nx-1,nz-1) ??
    # arithmetic mean for ρ
    @. ρ_ihalf_jhalf = (ρ[2:end,2:end] .+ ρ[2:end, 1:end-1] .+ ρ[1:end-1, 2:end] .+ ρ[1:end-1, 1:end-1]) ./ 4.0
    # μ_ihalf (nx-1,nz) ??
    # μ_ihalf (nx,nz-1) ??
    if harmonicaver_μ == true
        # harmonic mean for μ
        @. μ_ihalf = 1.0 ./ (1.0 ./ μ[2:end, :] .+ 1.0 ./ μ[1:end-1, :])
        @. μ_jhalf = 1.0 ./ (1.0 ./ μ[:, 2:end] .+ 1.0 ./ μ[:, 1:end-1])
    else
        # arithmetic mean for μ
        @. μ_ihalf = (μ[2:end, :] + μ[1:end-1, :]) / 2.0
        @. μ_jhalf = (μ[:, 2:end] + μ[:, 1:end-1]) / 2.0
    end
    # λ_ihalf (nx-1,nz) ??
    # arithmetic mean for λ
    @. λ_ihalf = (λ[2:end, :] + λ[1:end-1, :]) / 2.0

    return
end

##############################################################

struct ElasticIsoCPMLWaveSimulation{T, N, A <: AbstractArray{T, N}, V <: AbstractVector{T}} <: ElasticIsoWaveSimulation{T, N}
    # Parameters
    params::InputParametersElastic{T, N}
    cpmlparams::CPMLBoundaryConditionParameters{T}
    # Numerics
    nt::Int
    dt::T
    # Computational grid
    grid::UniformFiniteDifferenceGrid{N, T}
    # Logging parameters
    infoevery::Int
    # Material properties
    matprop::ElasticIsoMaterialProperties{T, N}
    # CPML coefficients
    cpmlcoeffs::NTuple{N, CPMLCoefficientsAxis{T, V}}
    # Checkpointing setup
    checkpointer::Union{Nothing, LinearCheckpointer{T}}
    # Smooth radius for gradient
    smooth_radius::Int
    # Snapshotter setup
    snapshotter::Union{Nothing, LinearSnapshotter{T, N, Array{T, N}}}
    # Parallelization type
    parall::Symbol

    function ElasticIsoCPMLWaveSimulation(
        params::InputParametersElastic{T, N},
        matprop::ElasticIsoMaterialProperties{T, N},
        cpmlparams::CPMLBoundaryConditionParameters{T};
        parall::Symbol=:threads,
        gradient::Bool=false,
        check_freq::Union{Int, Nothing}=nothing,
        snapevery::Union{Int, Nothing}=nothing,
        infoevery::Union{Int, Nothing}=nothing,
        smooth_radius::Int=5
    ) where {T, N}
        # Extract params
        nt = params.ntimesteps
        dt = params.dt
        gridspacing = params.gridspacing
        gridsize = params.gridsize
        halo = cpmlparams.halo
        freetop = cpmlparams.freeboundtop
        # Check BDC parameters
        @assert halo >= 0 "CPML halo size must be non-negative!"
        gridsize_cpml = freetop ? gridsize[1:(end-1)] : gridsize
        @assert all(n -> n >= 2halo + 3, gridsize_cpml) "Number grid points in the dimensions with C-PML boundaries must be at least 2*halo+3 = $(2halo+3)!"

        # Select backend
        backend = select_backend(ElasticIsoCPMLWaveSimulation{T, N}, parall)
        A = backend.Data.Array{T, N}
        V = backend.Data.Array{T, 1}
        # Initialize computational grid
        grid = UniformFiniteDifferenceGrid(gridsize, gridspacing)
        # Initialize CPML coefficients
        cpmlcoeffs = tuple([CPMLCoefficientsAxis{T, V}(halo, backend) for _ in 1:N]...)

        # Populate computational grid
        if N==2
            # Stress and velocity
            addfield!(grid, "σ" => MultiVariableField(
                [backend.zeros(T, gridsize...) for _ in 1:3]
            ))
            addfield!(grid, "v" => MultiVariableField(
                [backend.zeros(T, gridsize...) for _ in 1:2]
            ))
            # Material properties
            addfield!(grid, "λ" => ScalarVariableField(
                backend.zeros(T, gridsize...)
            ))
            addfield!(grid, "μ" => ScalarVariableField(
                backend.zeros(T, gridsize...)
            ))
            addfield!(grid, "ρ" => ScalarVariableField(
                backend.zeros(T, gridsize...)
            ))
            addfield!(grid, "λ_ihalf" => ScalarVariableField(
                backend.zeros(T, (gridsize.-[1,0])...)
            ))
            addfield!(grid, "μ_ihalf" => ScalarVariableField(
                backend.zeros(T, (gridsize.-[1,0])...)
            ))
            addfield!(grid, "μ_jhalf" => ScalarVariableField(
                backend.zeros(T, (gridsize.-[0,1])...)
            ))
            addfield!(grid, "ρ_ihalf_jhalf" => ScalarVariableField(
                backend.zeros(T, (gridsize.-1)...)
            ))
            # CPML memory variables
            grds = [gridsize...]
            gs1, gs2 = copy(grds), copy(grds)
            gs1[1] = 2 * halo
            gs2[2] = 2 * halo
            addfield!(grid, "ψ_∂σ∂x" => MultiVariableField(
                [backend.zeros(T, gs1...) for _ in 1:2]
            ))
            addfield!(grid, "ψ_∂σ∂z" => MultiVariableField(
                [backend.zeros(T, gs2...) for _ in 1:2]
            ))
            addfield!(grid, "ψ_∂v∂x" => MultiVariableField(
                [backend.zeros(T, gs1...) for _ in 1:2]
            ))
            addfield!(grid, "ψ_∂v∂z" => MultiVariableField(
                [backend.zeros(T, gs2...) for _ in 1:2]
            ))
        else
            error("Only elastic 2D is currently implemented.")
        end

        # Initialize CPML coefficients
        cpmlcoeffs = tuple([CPMLCoefficientsAxis{T, V}(halo, backend) for _ in 1:N]...)

        # Initialize gradient arrays if needed
        if gradient
            error("Gradient for elastic calculations not yet implemented!")
        end

        if snapevery !== nothing
            # Initialize snapshotter
            snapshotter = LinearSnapshotter{Array{T, N}}(nt, snapevery, Dict("v" => MultiVariableField(
                [backend.zeros(T, gridsize...) for _ in 1:N]
            )))
        end

        # Check infoevery
        if infoevery === nothing
            infoevery = nt + 2  # never reach it
        else
            @assert infoevery >= 1 && infoevery <= nt "Infoevery parameter must be positive and less then nt!"
        end
        

        return new{T, N, A, V}(
            params,
            cpmlparams,
            nt,
            dt,
            grid,
            infoevery,
            matprop,
            cpmlcoeffs,
            gradient ? checkpointer : nothing,
            smooth_radius,
            snapevery === nothing ? nothing : snapshotter,
            parall
        )
    end
end

###########################################################

# Specific functions for ElasticIsoCPMLWaveSimulation

@views function reset!(model::ElasticIsoCPMLWaveSimulation{T, N}) where {T, N}
    # Reset computational arrays
    reset!(model.grid; except=["ρ", "λ", "μ", "λ_ihalf", "μ_ihalf", "μ_jhalf", "ρ_ihalf_jhalf"])
    if model.checkpointer !== nothing
        reset!(model.checkpointer)
    end
end
###########################################################

# Traits for ElasticIsoCPMLWaveSimulation

IsSnappableTrait(::Type{<:ElasticIsoCPMLWaveSimulation}) = Snappable()
BoundaryConditionTrait(::Type{<:ElasticIsoCPMLWaveSimulation}) = CPMLBoundaryCondition()
GridTrait(::Type{<:ElasticIsoCPMLWaveSimulation}) = LocalGrid()

###########################################################

struct ElasticIsoReflWaveSimulation{T, N} <: ElasticIsoWaveSimulation{T, N} end    # TODO implementation

###########################################################

# Traits for ElasticIsoReflWaveSimulation

IsSnappableTrait(::Type{<:ElasticIsoReflWaveSimulation}) = Snappable()
BoundaryConditionTrait(::Type{<:ElasticIsoReflWaveSimulation}) = ReflectiveBoundaryCondition()
GridTrait(::Type{<:ElasticIsoReflWaveSimulation}) = LocalGrid()

###########################################################
