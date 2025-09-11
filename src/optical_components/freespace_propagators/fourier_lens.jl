function FourierLens(u::ScalarField{U, Nd},
        ds::NTuple{Nd, Real},
        ds′::NTuple{Nd, Real},
        fl::Real;
        use_cache::Bool = true,
        double_precision_kernel::Bool = true
) where {Nd, U <: AbstractArray{<:Complex}}
    CollinsProp(u, ds, ds′, (0, fl, 0); use_cache, double_precision_kernel)
end

function FourierLens(u::ScalarField{U, Nd},
        ds′::NTuple{Nd, Real},
        fl::Real;
        use_cache::Bool = true,
        double_precision_kernel::Bool = true
) where {Nd, U <: AbstractArray{<:Complex}}
    CollinsProp(u, ds′, (0, fl, 0); use_cache, double_precision_kernel)
end
