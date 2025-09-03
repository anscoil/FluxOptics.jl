const BPMProp = Union{Type{ASProp}, Type{TiltedASProp}}

struct BPM{M, A, U, D, P} <: AbstractCustomComponent{M}
    dn::A
    aperture_mask::D
    p_bpm::P
    p_bpm_half::P
    ∂p::Union{Nothing, @NamedTuple{dn::A}}
    u::Union{Nothing, U}

    function BPM(dn::A, aperture_mask::D, p_bpm::P, p_bpm_half::P,
            ∂p::Union{Nothing, @NamedTuple{dn::A}},
            u::U) where {A, D, P, U}
        M = Trainable{Buffered} # Only possibility to call this constructor
        new{M, A, U, D, P}(dn, aperture_mask, p_bpm, p_bpm_half, ∂p, u)
    end

    function _init(u::U, ds::NTuple{Nd, Real}, thickness::Real,
            dn0::AbstractArray{<:Real, Nv}, trainable::Bool,
            aperture::Function
    ) where {Nd, Nv, N, T, U <: AbstractArray{Complex{T}, N}}
        @assert Nd in (1, 2)
        @assert Nv == Nd + 1
        @assert N >= Nd
        n_slices = size(dn0, Nv)
        @assert n_slices >= 2
        dz = thickness / n_slices
        M = trainable ? Trainable{Buffered} : Static # This propagator requires buffering
        A = adapt_dim(U, Nv, real)
        D = adapt_dim(U, Nd, real)
        dn = A(dn0)
        xs = spatial_vectors(size(u)[1:Nd], ds)
        aperture_mask = Nd == 2 ? D(aperture.(xs[1], xs[2]')) : D(aperture.(xs[1]))
        ∂p = trainable ? (; dn = similar(dn)) : nothing
        u = trainable ? similar(u, (size(u)..., n_slices)) : nothing
        ((M, A, U, D, P), (dn, dz, aperture_mask, ∂p, u))
    end

    function BPM(Prop::BPMProp, u::AbstractArray{<:Complex}, ds::NTuple,
            thickness::Real, λ::Real, n0::Real, dn0::AbstractArray{<:Real};
            trainable::Bool = false,
            aperture::Function = (_...) -> 1,
            double_precision_kernel::Bool = true, args = (), kwargs = (;))
        ((M, A, U, D, P),
            (dn, dz, aperture_mask, ∂p, u)
        ) = _init(u, ds, thickness, dn0, trainable, aperture)
        p_bpm = Prop(u, ds, dz, args..., λ; n0, double_precision_kernel, kwargs...)
        p_bpm_half = Prop(u, ds, dz/2, args..., λ; n0, double_precision_kernel, kwargs...)
        P = typeof(p_bpm)
        new{M, A, U, D, P}(dn, aperture_mask, p_bpm, p_bpm_half, ∂p, u)
    end

    function BPM(Prop::BPMProp, use_cache::Bool, u::ScalarField, thickness::Real,
            n0::Real, dn0::AbstractArray{<:Real};
            trainable::Bool = false,
            aperture::Function = (_...) -> 1,
            double_precision_kernel::Bool = true, args = (), kwargs = (;))
        ((M, A, U, D, P),
            (dn, dz, aperture_mask, ∂p, u)
        ) = _init(u.data, u.ds, thickness, dn0, trainable, aperture)
        p_bpm = Prop(u, dz, args..., use_cache; n0, double_precision_kernel, kwargs...)
        p_bpm_half = Prop(u, dz/2, args..., use_cache;
            n0, double_precision_kernel, kwargs...)
        P = typeof(p_bpm)
        new{M, A, U, D, P}(dn, aperture_mask, p_bpm, p_bpm_half, ∂p, u)
    end
end

function AS_BPM(u::AbstractArray{<:Complex}, ds::NTuple,
        thickness::Real, λ::Real, n0::Real, dn0::AbstractArray{<:Real};
        trainable::Bool = false,
        aperture::Function = (_...) -> 1,
        double_precision_kernel::Bool = true)
    BPM(ASProp, u, ds, thickness, λ, n0, dn0; trainable, aperture, double_precision_kernel,
        kwargs = (; paraxial = true))
end

function AS_BPM(u::ScalarField, thickness::Real, n0::Real,
        dn0::AbstractArray{<:Real}, use_cache::Bool = false;
        trainable::Bool = false,
        aperture::Function = (_...) -> 1,
        double_precision_kernel::Bool = true)
    BPM(ASProp, use_cache, u, thickness, n0, dn0; trainable, aperture,
        double_precision_kernel, kwargs = (; paraxial = true))
end

function TiltedAS_BPM(u::AbstractArray{<:Complex}, ds::NTuple{Nd, Real}, thickness::Real,
        θs::NTuple{Nd, Union{Real, AbstractVector{<:Real}}}, λ::Real,
        n0::Real, dn0::AbstractArray{<:Real};
        trainable::Bool = false,
        aperture::Function = (_...) -> 1,
        double_precision_kernel::Bool = true) where {Nd}
    BPM(TiltedASProp, u, ds, thickness, λ, n0, dn0; trainable, aperture,
        double_precision_kernel, args = (θs,))
end

function TiltedAS_BPM(u::ScalarField, thickness::Real,
        θs::NTuple{Nd, Union{Real, AbstractVector{<:Real}}},
        n0::Real, dn0::AbstractArray{<:Real}, use_cache::Bool = false;
        trainable::Bool = false,
        aperture::Function = (_...) -> 1,
        double_precision_kernel::Bool = true) where {Nd}
    BPM(TiltedASProp, use_cache, u, thickness, n0, dn0; trainable, aperture,
        double_precision_kernel, args = (θs,))
end

Functors.@functor BPM (dn,)

function Base.collect(p::BPM)
    if ndims(p.dn) == 2
        collect(p.dn)
    else
        [collect(dn) for dn in reshape(eachslice(p.dn, dims = 3), :)]
    end
end

function Base.fill!(p::BPM, v::Real)
    p.dn .= v
    p
end

function Base.fill!(p::BPM, v::AbstractArray)
    copyto!(p.dn, v)
    p
end

trainable(p::BPM{<:Trainable}) = (; dn = p.dn)

get_preallocated_gradient(p::BPM{Trainable{Buffered}}) = p.∂p

get_saved_buffer(p::BPM{Trainable{Buffered}}) = p.u
