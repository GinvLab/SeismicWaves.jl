zeros(T, x...) = @zeros(x..., eltype = T)
ones(T, x...) = @ones(x..., eltype = T)
rand(T, x...) = @rand(x..., eltype = T)
fill(T, x...) = @rand(x..., eltype = T)