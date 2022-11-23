# comparison between two BSON dictionaries, key per key
# using ≈ (a.k.a. isapprox() ) to compare matrices, which compares their LinearAlgebra.norm()
# to add tolerance, one could use the following lines:
# isapprox(v1,v2;atol=ϵ) # for absolute tolerance of ϵ
# isapprox(v1,v2;rtol=ϵ) # for relative tolerance of ϵ
comp(d1, d2) = keys(d1)==keys(d2) && all([ v1 ≈ v2 for (v1,v2) in zip(values(d1), values(d2))])
