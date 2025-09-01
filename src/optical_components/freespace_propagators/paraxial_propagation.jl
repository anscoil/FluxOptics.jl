# Edward A. Sziklas and A. E. Siegman, "Mode calculations in unstable
# resonators with flowing saturable gain. 2: Fast Fourier transform
# method," Appl. Opt. 14, 1874-1889 (1975)
# https://doi.org/10.1364/AO.14.001874

# Careful that z != 0
function ParaxialProp(u::AbstractArray{<:Complex},
        ds::NTuple{Nd, Real},
        ds′::NTuple{Nd, Real},
        z::Real,
        λ::Real;
        double_precision_kernel::Bool = true
) where {Nd}
    CollinsProp(
        u, ds, ds′, (1, z, 1), λ; double_precision_kernel = double_precision_kernel)
end

# Careful that z != 0
function ParaxialProp(u::ScalarField{U, Nd},
        ds::NTuple{Nd, Real},
        ds′::NTuple{Nd, Real},
        z::Real,
        use_cache::Bool = false;
        filter = nothing,
        double_precision_kernel::Bool = true
) where {Nd, U <: AbstractArray{<:Complex}}
    if ds == ds′
        ASProp(u, ds, z, use_cache;
            filter = filter, paraxial = true,
            double_precision_kernel = double_precision_kernel)
    else
        CollinsProp(u, ds, ds′, (1, z, 1), use_cache;
            double_precision_kernel = double_precision_kernel)
    end
end

# Careful that z != 0
function ParaxialProp(u::ScalarField{U, Nd},
        ds′::NTuple{Nd, Real},
        z::Real,
        use_cache::Bool = false;
        filter = nothing,
        double_precision_kernel::Bool = true
) where {Nd, U <: AbstractArray{<:Complex}}
    ParaxialProp(u, u.ds, ds′, z, use_cache;
        double_precision_kernel = double_precision_kernel)
end

function ParaxialProp(u::AbstractArray{<:Complex},
        ds::NTuple{Nd, Real},
        z::Real,
        λ::Real;
        filter = nothing,
        double_precision_kernel::Bool = true
) where {Nd}
    ASProp(u, ds, z, λ;
        filter = filter, paraxial = true,
        double_precision_kernel = double_precision_kernel)
end

function ParaxialProp(u::ScalarField,
        z::Real,
        use_cache::Bool = false;
        filter = nothing,
        double_precision_kernel::Bool = true)
    ASProp(u, z, use_cache;
        filter = filter, paraxial = true,
        double_precision_kernel = double_precision_kernel)
end
