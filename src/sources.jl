"""
Type representing a multi-source configuration for a wave propagation shot.
"""
struct ScalarSources{T <: Real} <: Sources
    positions::Matrix{<:Real}
    tf::Matrix{T}
    domfreq::T

    @doc """
        ScalarSources{T<:Real}( 
            positions::Matrix{<:Real},
            tf::Matrix{<:T},
            domfreq::T
        )

    Create a single shot wave propagation source configuration from source positions, time-functions and a dominant frequency.
    """
    function ScalarSources{T}(positions::Matrix{<:Real}, tf::Matrix{T}, domfreq::T) where {T <: Real}
        @assert size(positions, 1) > 0 "There must be at least one source!"
        @assert size(positions, 1) == size(tf, 2) "Number of sources do not match between positions and time-functions!"
        return new(positions, tf, domfreq)
    end
end

# Default type constructor {Float64}
ScalarSources(positions, tf, domfreq) = ScalarSources{Float64}(positions, tf, domfreq)

####################################################33

"""
Type representing a 2D moment tensor.
"""
struct MomentTensor2D{T <: Real}
  Mxx::Vector{T}
  Mzz::Vector{T}
  Mxz::Vector{T}
end


"""
Type representing a multi-source configuration for a wave propagation shot.
"""
struct MomentTensor2DSources{T <: Real} <: Sources
    positions::Matrix{<:Real}
    tf::Matrix{T}
    momtens::MomentTensor2D
    domfreq::T

    @doc """
        MomentTensorSources{T<:Real}( 
            positions::Matrix{<:Real},
            tf::Matrix{<:T},
            momtens::MomentTensor2D
            domfreq::T
        )

    Create a single shot wave propagation source configuration from source positions, time-functions and a dominant frequency.
    """
    function MomentTensor2DSources{T}(positions::Matrix{<:Real}, tf::Matrix{T},  momtens::MomentTensor2D, domfreq::T) where {T <: Real}
        @assert size(positions, 1) > 0 "There must be at least one source!"
        @assert size(positions, 1) == size(tf, 2) "Number of sources do not match between positions and time-functions!"
        return new(positions, tf, momtens, domfreq)
    end
end

# Default type constructor {Float64}
MomentTensorSources(positions, tf, momtens, domfreq) = MomentTensorSources{Float64}(positions, tf, momtens, domfreq)

####################################################
