"""
    @views function forward!(
        model::WaveModel,
        srcs::Sources{<:Real},
        recs::Receivers{<:Real}
    )

Compute the forward wave propagation on the specified `WaveModel` and sources/receivers configurations.
The result of the computation is saved in the `WaveModel`.
"""
@views function forward!(
    model::WaveModel,
    srcs::Sources{<:Real},
    recs::Receivers{<:Real}
)

## TODO

end