function FourierLens(u::AbstractArray{<:Complex},
        ds::NTuple{Nd, Real},
        ds′::NTuple{Nd, Real},
        fl::Real,
        λ::Real;
        double_precision_kernel::Bool = true
) where {Nd}
    CollinsProp(
        u, ds, ds′, (0, fl, 0), λ; double_precision_kernel = double_precision_kernel)
end

function FourierLens(u::ScalarField{U},
        ds::NTuple{Nd, Real},
        ds′::NTuple{Nd, Real},
        fl::Real,
        use_cache::Bool = false;
        double_precision_kernel::Bool = true
) where {Nd, U <: AbstractArray{<:Complex}}
    CollinsProp(u, ds, ds′, (0, fl, 0), use_cache;
        double_precision_kernel = double_precision_kernel)
end
