

# -*- coding: utf-8 -*-
#
# Andrea Zunino
# 
# 
##==========================================================

module ElasticWaves



export InpParam,RockProperties,gaussource1D,rickersource1D
export solveelastic2D_reflbound,solveelastic2D_CPML,MomentTensor

include("elasticwaveprop_sincinterp.jl")



end # module
