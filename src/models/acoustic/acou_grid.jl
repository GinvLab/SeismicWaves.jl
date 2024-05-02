struct AcousticCDForwardGrid{N, T, A <: AbstractArray{T,N}, V <: AbstractVector{T}}
    fact::A
    pold::A
    pcur::A
    pnew::A
    ψ::Vector{A}
    ξ::Vector{A}
    a_coeffs::Vector{V}
    b_coeffs::Vector{V}

    function AcousticCDForwardGrid(ns::NTuple{N, Int}, halo::Int, cpmlcoeffs::NTuple{N, CPMLCoefficients}, backend::Module, T::Type) where {N}
        A = backend.Data.Array{N}
        V = backend.Data.Array{1}
        # Initialize computational arrays
        fact = backend.zeros(ns...)
        pold = backend.zeros(ns...)
        pcur = backend.zeros(ns...)
        pnew = backend.zeros(ns...)
        # Initialize CPML arrays
        ψ = []
        ξ = []
        for i in 1:N
            ψ_ns = [ns...]
            ξ_ns = [ns...]
            ψ_ns[i] = halo + 1
            ξ_ns[i] = halo
            append!(ψ, [backend.zeros(ψ_ns...), backend.zeros(ψ_ns...)])
            append!(ξ, [backend.zeros(ξ_ns...), backend.zeros(ξ_ns...)])
        end
        # Build CPML coefficient arrays for computations (they are just references to cpmlcoeffs)
        a_coeffs = []
        b_coeffs = []
        for i in 1:N
            append!(a_coeffs, [cpmlcoeffs[i].a_l, cpmlcoeffs[i].a_r, cpmlcoeffs[i].a_hl, cpmlcoeffs[i].a_hr])
            append!(b_coeffs, [cpmlcoeffs[i].b_l, cpmlcoeffs[i].b_r, cpmlcoeffs[i].b_hl, cpmlcoeffs[i].b_hr])
        end

        new{N,T,A,V}(fact, pold, pcur, pnew, ψ, ξ, a_coeffs, b_coeffs)
    end
end

struct AcousticCDAdjointGrid{N, T, A <: AbstractArray{T,N}}
    curgrad::A
    adjold::A
    adjcur::A
    adjnew::A
    ψ_adj::Vector{A}
    ξ_adj::Vector{A}
    totgrad_size::Vector{Int}
    smooth_radius::Int

    function AcousticCDAdjointGrid(ns::NTuple{N, Int}, halo::Int, backend::Module, smooth_radius::Int, T::Type) where {N}
        A = backend.Data.Array{N}

        totgrad_size = [ns...]
        # Current gradient array
        curgrad = backend.zeros(ns...)
        # Adjoint arrays
        adjold = backend.zeros(ns...)
        adjcur = backend.zeros(ns...)
        adjnew = backend.zeros(ns...)
        # Initialize CPML arrays
        ψ_adj = []
        ξ_adj = []
        for i in 1:N
            ψ_ns = [ns...]
            ξ_ns = [ns...]
            ψ_ns[i] = halo + 1
            ξ_ns[i] = halo
            append!(ψ_adj, [backend.zeros(ψ_ns...), backend.zeros(ψ_ns...)])
            append!(ξ_adj, [backend.zeros(ξ_ns...), backend.zeros(ξ_ns...)])
        end

        new{N,T,A}(curgrad, adjold, adjcur, adjnew, ψ_adj, ξ_adj, totgrad_size, smooth_radius)
    end
end

struct AcousticCDCheckpointing{N, T, A <: AbstractArray{T,N}}
    check_freq::Union{Int, Nothing}
    last_checkpoint::Int
    save_buffer::Vector{A}
    checkpoints::Dict{Int, A}
    checkpoints_ψ::Dict{Int, Vector{A}}
    checkpoints_ξ::Dict{Int, Vector{A}}

    function AcousticCDCheckpointing(
        check_freq::Union{Int, Nothing},
        nt::Int,
        ns::NTuple{N, Int},
        forward_grid::AcousticCDForwardGrid{N},
        backend::Module,
        T::Type
    ) where {N}
        A = backend.Data.Array{N}
        if check_freq !== nothing
            @assert check_freq > 2 "Checkpointing frequency must be bigger than 2!"
            @assert check_freq < nt "Checkpointing frequency must be smaller than the number of timesteps!"
            # Time step of last checkpoint
            last_checkpoint = floor(Int, nt / check_freq) * check_freq
            # Checkpointing arrays
            save_buffer = [backend.zeros(ns...) for _ in 1:(check_freq + 2)]      # pressure window buffer
            checkpoints = Dict{Int, A}()           # pressure checkpoints
            checkpoints_ψ = Dict{Int, Vector{A}}()                        # ψ arrays checkpoints
            checkpoints_ξ = Dict{Int, Vector{A}}()                        # ξ arrays checkpoints
            # Save initial conditions as first checkpoint
            checkpoints[-1] = copy(forward_grid.pold)
            checkpoints[0] = copy(forward_grid.pcur)
            checkpoints_ψ[0] = copy.(forward_grid.ψ)
            checkpoints_ξ[0] = copy.(forward_grid.ξ)
            # Preallocate future checkpoints
            for it in 1:(nt+1)
                if it % check_freq == 0
                    checkpoints[it] = backend.zeros(ns...)
                    checkpoints[it-1] = backend.zeros(ns...)
                    checkpoints_ψ[it] = copy.(forward_grid.ψ)
                    checkpoints_ξ[it] = copy.(forward_grid.ξ)
                end
            end
        else    # no checkpointing
            last_checkpoint = 0                                 # simulate a checkpoint at time step 0 (so buffer will start from -1)
            save_buffer = [backend.zeros(ns...) for _ in 1:(nt + 2)]    # save all timesteps (from -1 to nt+1 so nt+2)
            checkpoints = Dict{Int, backend.Data.Array}()       # pressure checkpoints (will remain empty)
            checkpoints_ψ = Dict{Int, Any}()                    # ψ arrays checkpoints (will remain empty)
            checkpoints_ξ = Dict{Int, Any}()                    # ξ arrays checkpoints (will remain empty)
        end
        # Save first 2 timesteps in save buffer
        copyto!(save_buffer[1], forward_grid.pold)
        copyto!(save_buffer[2], forward_grid.pcur)

        new{N,T,A}(check_freq, last_checkpoint, save_buffer, checkpoints, checkpoints_ψ, checkpoints_ξ)
    end
end