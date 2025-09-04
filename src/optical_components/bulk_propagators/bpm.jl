const BPMProp = Union{Type{ASProp}, Type{TiltedASProp}}

compute_cos_correction(u, p) = 1

function compute_cos_correction(u::AbstractArray, p::TiltedASProp)
    reduce((.*), map(x -> cos.(x), p.θs))
end

struct BPM{M, A, U, D, P, K} <: AbstractCustomComponent{M}
    dn::A
    kdz::K
    aperture_mask::D
    p_bpm::P
    p_bpm_half::P
    ∂p::Union{Nothing, @NamedTuple{dn::A}}
    u::Union{Nothing, U}

    function BPM(dn::A, kdz::K, aperture_mask::D, p_bpm::P, p_bpm_half::P,
            ∂p::Union{Nothing, @NamedTuple{dn::A}},
            u::U) where {A, K, D, P, U}
        M = isnothing(u) ? Trainable{Unbuffered} : Trainable{Buffered}
        new{M, A, U, D, P, K}(dn, kdz, aperture_mask, p_bpm, p_bpm_half, ∂p, u)
    end

    function _init(u::U, ds::NTuple{Nd, Real}, thickness::Real,
            dn0::AbstractArray{<:Real, Nv}, trainable::Bool, buffered::Bool,
            aperture::Function
    ) where {Nd, Nv, N, T, U <: AbstractArray{Complex{T}, N}}
        @assert Nd in (1, 2)
        @assert Nv == Nd + 1
        @assert N >= Nd
        n_slices = size(dn0, Nv)
        @assert n_slices >= 2
        dz = thickness / n_slices
        M = trainability(trainable, buffered)
        A = adapt_dim(U, Nv, real)
        D = adapt_dim(U, Nd, real)
        dn = A(dn0)
        xs = spatial_vectors(size(u)[1:Nd], ds)
        aperture_mask = Nd == 2 ? D(aperture.(xs[1], xs[2]')) : D(aperture.(xs[1]))
        ∂p = (trainable && buffered) ? (; dn = similar(dn)) : nothing
        u = (trainable && buffered) ? similar(u, (size(u)..., n_slices)) : nothing
        ((M, A, U, D, P), (dn, dz, aperture_mask, ∂p, u))
    end

    function BPM(Prop::BPMProp, u::AbstractArray{<:Complex}, ds::NTuple,
            thickness::Real, λ::Real, n0::Real, dn0::AbstractArray{<:Real};
            trainable::Bool = false, buffered::Bool = false,
            aperture::Function = (_...) -> 1,
            double_precision_kernel::Bool = true, args = (), kwargs = (;))
        ((M, A, U, D, P),
            (dn, dz, aperture_mask, ∂p, u)
        ) = _init(u, ds, thickness, dn0, trainable, buffered, aperture)
        p_bpm = Prop(u, ds, dz, args..., λ; n0, double_precision_kernel, kwargs...)
        p_bpm_half = Prop(u, ds, dz/2, args..., λ; n0, double_precision_kernel, kwargs...)
        P = typeof(p_bpm)
        kdz = (2π/λ*dz) ./ compute_cos_correction(u, p_bpm)
        K = typeof(kdz)
        new{M, A, U, D, P, K}(dn, kdz, aperture_mask, p_bpm, p_bpm_half, ∂p, u)
    end

    function BPM(Prop::BPMProp, use_cache::Bool, u::ScalarField, thickness::Real,
            n0::Real, dn0::AbstractArray{<:Real};
            trainable::Bool = false, buffered::Bool = false,
            aperture::Function = (_...) -> 1,
            double_precision_kernel::Bool = true, args = (), kwargs = (;))
        ((M, A, U, D, P),
            (dn, dz, aperture_mask, ∂p, u)
        ) = _init(u.data, u.ds, thickness, dn0, trainable, buffered, aperture)
        p_bpm = Prop(u, dz, args..., use_cache; n0, double_precision_kernel, kwargs...)
        p_bpm_half = Prop(u, dz/2, args..., use_cache;
            n0, double_precision_kernel, kwargs...)
        P = typeof(p_bpm)
        kdz = (2π*dz) ./ compute_cos_correction(u.data, p_bpm)
        K = typeof(kdz)
        new{M, A, U, D, P, K}(dn, kdz, aperture_mask, p_bpm, p_bpm_half, ∂p, u)
    end
end

function AS_BPM(u::AbstractArray{<:Complex}, ds::NTuple,
        thickness::Real, λ::Real, n0::Real, dn0::AbstractArray{<:Real};
        trainable::Bool = false, buffered::Bool = false,
        aperture::Function = (_...) -> 1,
        double_precision_kernel::Bool = true)
    BPM(ASProp, u, ds, thickness, λ, n0, dn0; trainable, buffered, aperture,
        double_precision_kernel, kwargs = (; paraxial = true))
end

function AS_BPM(u::ScalarField, thickness::Real, n0::Real,
        dn0::AbstractArray{<:Real}, use_cache::Bool = false;
        trainable::Bool = false, buffered::Bool = false,
        aperture::Function = (_...) -> 1,
        double_precision_kernel::Bool = true)
    BPM(ASProp, use_cache, u, thickness, n0, dn0; trainable, buffered, aperture,
        double_precision_kernel, kwargs = (; paraxial = true))
end

function TiltedAS_BPM(u::AbstractArray{<:Complex}, ds::NTuple{Nd, Real}, thickness::Real,
        θs::NTuple{Nd, Union{Real, AbstractVector{<:Real}}}, λ::Real,
        n0::Real, dn0::AbstractArray{<:Real};
        trainable::Bool = false,
        aperture::Function = (_...) -> 1,
        double_precision_kernel::Bool = true) where {Nd}
    cos_correction = compute_cos_correction(u, θs)
    BPM(TiltedASProp, u, ds, thickness, λ, n0, dn0; trainable, aperture,
        double_precision_kernel, args = (θs,))
end

function TiltedAS_BPM(u::ScalarField, thickness::Real,
        θs::NTuple{Nd, Union{Real, AbstractVector{<:Real}}},
        n0::Real, dn0::AbstractArray{<:Real}, use_cache::Bool = false;
        trainable::Bool = false,
        aperture::Function = (_...) -> 1,
        double_precision_kernel::Bool = true) where {Nd}
    cos_correction = compute_cos_correction(u, θs)
    BPM(TiltedASProp, use_cache, u, thickness, n0, dn0; trainable, aperture,
        double_precision_kernel, args = (θs,))
end

Functors.@functor BPM (dn,)

function Base.collect(p::BPM)
    if ndims(p.dn) == 2
        collect(p.dn)
    else
        [collect(dn) for dn in eachslice(p.dn, dims = 3)]
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

function alloc_saved_buffer(u, p::BPM{Trainable{Unbuffered}})
    Nv = ndims(p.dn)
    n_slices = size(p.dn, Nv)
    u_data = get_data(u)
    similar(u_data, (size(u_data)..., n_slices))
end

get_saved_buffer(p::BPM{Trainable{Buffered}}) = p.u

function apply_dn_slice!(u::ScalarField, dn::AbstractArray, kdz,
        direction::Type{<:Direction})
    s = sign(direction)
    @. u.data *= cis(s*kdz/u.lambdas*dn)
end

function apply_dn_slice!(u::AbstractArray, dn::AbstractArray, kdz,
        direction::Type{<:Direction})
    s = sign(direction)
    @. u *= cis(s*kdz*dn)
end

function propagate!(u, p::BPM, direction::Type{<:Direction}; u_saved = nothing)
    Nv = ndims(p.dn)
    n_slices = size(p.dn, Nv)
    dn_slices = eachslice(p.dn, dims = Nv)
    u_saved_slices = isnothing(u_saved) ?
                     Iterators.cycle(nothing) : eachslice(u_saved, dims = Nv)
    propagate!(u, p.p_bpm_half, direction)
    for (dn, u_saved) in zip(@view(dn_slices[1:(end - 1)]), u_saved_slices)
        copyto!(u_saved, get_data(u))
        apply_dn_slice!(u, dn, p.kdz, direction)
        propagate!(u, p.p_bpm, direction)
    end
    copyto!(u_saved_slices[end], get_data(u))
    apply_dn_slice!(u, dn_slices[end], p.kdz, direction)
    propagate!(u, p.p_bpm_half, direction)
    u
end

function backpropagate!(u, p::BPM, direction::Type{<:Direction})
    # propagate!(u, p, reverse(direction))
end

function propagate_and_save!(u, p::BPM{Trainable{Buffered}}, direction::Type{<:Direction})
    propagate!(u, p, direction; u_saved = p.u)
end

function propagate_and_save!(u, u_saved, p::BPM{Trainable{Unbuffered}},
        direction::Type{<:Direction})
    propagate!(u, p, direction; u_saved)
end

function backpropagate_with_gradient!(∂v, u_saved, ∂p::NamedTuple,
        p::BPM{<:Trainable}, direction::Type{<:Direction})
    # ∂u = backpropagate!(∂v, p, direction)
    # compute_phase_gradient!(∂p.ϕ, get_data(∂u), get_data(u_saved))
    # (∂u, ∂p)
end
