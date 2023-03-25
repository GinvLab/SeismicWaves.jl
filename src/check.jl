"""
    check(model::WaveSimul)

Check model for possible assertions based on configuration and model traits.
"""
#check(model::WaveSimul) = check(WaveEquationTrait(model), model)

function check(model::WaveSimul)
    @debug "Checking CFL condition"
    check_courant_condition(model)
end
