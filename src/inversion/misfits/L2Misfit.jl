
struct L2Misfit{T, D, OA <: AbstractArray{T, D}, IM <: AbstractMatrix{T}} <: AbstractMisfit{T}
    observed::OA
    invcov::IM
    windows::Vector{Pair{Int, Int}}

    function L2Misfit(; observed::AbstractArray{T, D}=zeros(T, 0, 0),
                        invcov::AbstractMatrix{T}=zeros(T, 0, 0),
                        windows::Vector{Pair{Int, Int}}=Vector{Pair{Int, Int}}()
    ) where {T, D}
        if !(D == 2 || D == 3)
            throw(ArgumentError("Observed data must be a 2D or 3D array!"))
        end
        @assert size(invcov, 1) == size(invcov, 2) "Inverse covariance matrix must be square!"
        @assert size(observed, 1) == size(invcov, 1) "Size of inverse covariance matrix must match the number of timesteps!"
        @assert all(w -> 1 <= w.first <= w.second <= size(observed, 1), windows) "Windows indices must be between 1 and maximum number of timesteps!"

        return new{T, D, typeof(observed), typeof(invcov)}(observed, invcov, windows)
    end
end



function calcmisfit(shotmisfit::L2Misfit{T, 2},recs::ScalarReceivers{T}) where {T}
    # Compute residuals
    residuals = recs.seismograms - shotmisfit.observed
    # Window residuals using mask
    mask = ones(T, size(residuals, 1))
    if length(shotmisfit.windows) > 0
        for wnd in  shotmisfit.windows
            mask[wnd.first:wnd.second] .= 2
        end
        mask .-= 1
    end
    residuals .= mask .* residuals
    # Inner product using inverse covariance matrix
    msf = dot(residuals,  shotmisfit.invcov, residuals) / 2
    # # Add regularization if needed
    # if misfit.regularization !== nothing
    #     msf += misfit.regularization(matprop)
    # end
    return msf
end

function calcmisfit(shotmisfit::L2Misfit{T, 3}, recs::VectorReceivers{T, N}) where {T, N}
    # Compute residuals
    residuals = recs.seismograms - shotmisfit.observed
    # Window residuals using mask
    mask = ones(T, size(residuals, 1))
    if length(shotmisfit.windows) > 0
        for wnd in shotmisfit.windows
            mask[wnd.first:wnd.second] .= 2
        end
        mask .-= 1
    end
    residuals .= mask .* residuals
    # Inner product using inverse covariance matrix
    msf = sum([dot(residuals[:, i, :], shotmisfit.invcov, residuals[:, i, :]) for i in 1:N]) / 2
    # Add regularization if needed
    # if misfit.regularization !== nothing
    #     msf += misfit.regularization(matprop)
    # end
    return msf
end

function ∂χ_∂u(shotmisfit::L2Misfit{T, 2}, recs::ScalarReceivers{T}) where {T}
    # Compute residuals
    residuals = recs.seismograms - shotmisfit.observed
    # Window residuals using mask
    mask = ones(T, size(residuals, 1))
    if length(shotmisfit.windows) > 0
        for wnd in shotmisfit.windows
            mask[wnd.first:wnd.second] .= 2
        end
        mask .-= 1
    end
    residuals .= mask .* residuals
    # Multiply with inverse of covariance matrix
    return shotmisfit.invcov * residuals
end

function ∂χ_∂u(shotmisfit::L2Misfit{T, 3}, recs::VectorReceivers{T}) where {T}
    # Compute residuals
    residuals = recs.seismograms - shotmisfit.observed
    # Window residuals using mask
    mask = ones(T, size(residuals, 1))
    if length(shotmisfit.windows) > 0
        for wnd in shotmisfit.windows
            mask[wnd.first:wnd.second] .= 2
        end
        mask .-= 1
    end
    residuals .= mask .* residuals
    # Multiply with inverse of covariance matrix
    for d in axes(residuals, 2)
        residuals[:, d, :] .= shotmisfit.invcov * residuals[:, d, :]
    end
    return residuals
end
