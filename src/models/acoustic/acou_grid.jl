# struct AcousticCDCheckpointing{N, T, A <: AbstractArray{T,N}}
#     check_freq::Union{Int, Nothing}
#     last_checkpoint::Int
#     save_buffer::Vector{A}
#     checkpoints::Dict{Int, A}
#     checkpoints_ψ::Dict{Int, Vector{A}}
#     checkpoints_ξ::Dict{Int, Vector{A}}

#     function AcousticCDCheckpointing(
#         check_freq::Union{Int, Nothing},
#         nt::Int,
#         ns::NTuple{N, Int},
#         grid::UniformFiniteDifferenceGrid{N,T},
#         backend::Module
#     ) where {N,T,A}
#         if check_freq !== nothing
#             @assert check_freq > 2 "Checkpointing frequency must be bigger than 2!"
#             @assert check_freq < nt "Checkpointing frequency must be smaller than the number of timesteps!"
#             # Time step of last checkpoint
#             last_checkpoint = floor(Int, nt / check_freq) * check_freq
#             # Checkpointing arrays
#             save_buffer = [backend.zeros(ns...) for _ in 1:(check_freq + 2)]      # pressure window buffer
#             checkpoints = Dict{Int, A}()           # pressure checkpoints
#             checkpoints_ψ = Dict{Int, Vector{A}}()                        # ψ arrays checkpoints
#             checkpoints_ξ = Dict{Int, Vector{A}}()                        # ξ arrays checkpoints
#             # Save initial conditions as first checkpoint
#             checkpoints[-1] = copy(grid.fields["pold"])
#             checkpoints[0] = copy(grid.fields["pcur"])
#             checkpoints_ψ[0] = copy.(forward_grid.ψ)
#             checkpoints_ξ[0] = copy.(forward_grid.ξ)
#             # Preallocate future checkpoints
#             for it in 1:(nt+1)
#                 if it % check_freq == 0
#                     checkpoints[it] = backend.zeros(ns...)
#                     checkpoints[it-1] = backend.zeros(ns...)
#                     checkpoints_ψ[it] = copy.(forward_grid.ψ)
#                     checkpoints_ξ[it] = copy.(forward_grid.ξ)
#                 end
#             end
#         else    # no checkpointing
#             last_checkpoint = 0                                 # simulate a checkpoint at time step 0 (so buffer will start from -1)
#             save_buffer = [backend.zeros(ns...) for _ in 1:(nt + 2)]    # save all timesteps (from -1 to nt+1 so nt+2)
#             checkpoints = Dict{Int, backend.Data.Array}()       # pressure checkpoints (will remain empty)
#             checkpoints_ψ = Dict{Int, Any}()                    # ψ arrays checkpoints (will remain empty)
#             checkpoints_ξ = Dict{Int, Any}()                    # ξ arrays checkpoints (will remain empty)
#         end
#         # Save first 2 timesteps in save buffer
#         copyto!(save_buffer[1], forward_grid.pold)
#         copyto!(save_buffer[2], forward_grid.pcur)

#         new{N,T,A}(check_freq, last_checkpoint, save_buffer, checkpoints, checkpoints_ψ, checkpoints_ξ)
#     end
# end