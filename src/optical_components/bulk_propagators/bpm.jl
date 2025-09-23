const BPMProp = Union{typeof(ASProp), typeof(TiltedASProp), typeof(ShiftProp)}

function compute_cos_correction(u::ScalarField)
    θs = get_tilts(u)
    reduce((.*), map(x -> cos.(x), θs))
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
        A = similar(U, real, Nv)
        D = similar(U, real, Nd)
        dn = A(dn0)
        xs = spatial_vectors(size(u)[1:Nd], ds)
        aperture_mask = Nd == 2 ? D(aperture.(xs[1], xs[2]')) : D(aperture.(xs[1]))
        ∂p = (trainable && buffered) ? (; dn = similar(dn)) : nothing
        u = (trainable && buffered) ? similar(u, (size(u)..., n_slices)) : nothing
        Us = similar(U, N+1)
        ((M, A, Us, D), (dn, dz, aperture_mask, ∂p, u))
    end

    function BPM(Prop::BPMProp, use_cache::Bool, u::ScalarField, thickness::Real,
            dn0::AbstractArray{<:Real};
            trainable::Bool = false, buffered::Bool = false,
            aperture::Function = (_...) -> 1,
            double_precision_kernel::Bool = use_cache, args = (), kwargs = (;))
        ((M, A, U, D),
            (dn, dz, aperture_mask, ∂p, u_saved)
        ) = _init(u.data, u.ds, thickness, dn0, trainable, buffered, aperture)
        p_bpm = Prop(u, dz, args...; use_cache, double_precision_kernel, kwargs...)
        p_bpm_half = Prop(u, dz/2, args...;
            use_cache, double_precision_kernel, kwargs...)
        P = typeof(p_bpm)
        kdz = (2π*dz) ./ compute_cos_correction(u)
        K = typeof(kdz)
        new{M, A, U, D, P, K}(dn, kdz, aperture_mask, p_bpm, p_bpm_half, ∂p, u_saved)
    end
end

function AS_BPM(u::ScalarField, thickness::Real, n0::Real,
        dn0::AbstractArray{<:Real}; use_cache::Bool = true,
        trainable::Bool = false, buffered::Bool = false,
        aperture::Function = (_...) -> 1,
        double_precision_kernel::Bool = use_cache)
    BPM(ASProp, use_cache, u, thickness, dn0; trainable, buffered, aperture,
        double_precision_kernel, kwargs = (; paraxial = true, n0))
end

function TiltedAS_BPM(u::ScalarField, thickness::Real,
        n0::Real, dn0::AbstractArray{<:Real}; use_cache::Bool = true,
        trainable::Bool = false, buffered::Bool = false,
        aperture::Function = (_...) -> 1,
        double_precision_kernel::Bool = use_cache)
    BPM(TiltedASProp, use_cache, u, thickness, dn0; trainable, buffered, aperture,
        double_precision_kernel, kwargs = (; n0))
end

function Shift_BPM(u::ScalarField, thickness::Real, dn0::AbstractArray{<:Real};
        use_cache::Bool = true, trainable::Bool = false, buffered::Bool = false,
        aperture::Function = (_...) -> 1,
        double_precision_kernel::Bool = use_cache)
    BPM(ShiftProp, use_cache, u, thickness, dn0; trainable, buffered, aperture,
        double_precision_kernel)
end

Functors.@functor BPM (dn,)

get_data(p::BPM) = p.dn

trainable(p::BPM{<:Trainable}) = (; dn = p.dn)

get_preallocated_gradient(p::BPM{Trainable{Buffered}}) = p.∂p

function alloc_saved_buffer(u::ScalarField, p::BPM{Trainable{Unbuffered}})
    Nv = ndims(p.dn)
    n_slices = size(p.dn, Nv)
    similar(u.data, (size(u.data)..., n_slices))
end

get_saved_buffer(p::BPM{Trainable{Buffered}}) = p.u

function apply_dn_slice!(u::ScalarField, dn::AbstractArray, kdz,
        direction::Type{<:Direction})
    s = sign(direction)
    lambdas = get_lambdas(u)
    @. u.data *= cis(s*kdz/lambdas*dn)
end

function propagate!(u::ScalarField, p::BPM, direction::Type{<:Direction}; u_saved = nothing)
    Nv = ndims(p.dn)
    n_slices = size(p.dn, Nv)
    dn_slices = eachslice(p.dn, dims = Nv)
    u_saved_slices = isnothing(u_saved) ?
                     Iterators.cycle(nothing) : eachslice(u_saved, dims = ndims(u_saved))
    propagate!(u, p.p_bpm_half, direction)
    for (dn, u_saved) in zip(@view(dn_slices[1:(end - 1)]), u_saved_slices)
        copyto!(u_saved, u.data)
        apply_dn_slice!(u, dn, p.kdz, direction)
        propagate!(u, p.p_bpm, direction)
    end
    copyto!(u_saved_slices[end], u.data)
    apply_dn_slice!(u, dn_slices[end], p.kdz, direction)
    propagate!(u, p.p_bpm_half, direction)
    u
end

function propagate_and_save!(u::ScalarField, p::BPM{Trainable{Buffered}},
        direction::Type{<:Direction})
    propagate!(u, p, direction; u_saved = p.u)
end

function propagate_and_save!(u::ScalarField, u_saved::AbstractArray,
        p::BPM{Trainable{Unbuffered}}, direction::Type{<:Direction})
    propagate!(u, p, direction; u_saved)
end

function compute_dn_gradient!(∂dn::AbstractArray{T, Nd}, u_saved, ∂u::ScalarField,
        kdz, direction) where {T <: Real, Nd}
    sdims = (Nd + 1):ndims(∂u)
    s = sign(direction)
    lambdas = get_lambdas(∂u)
    g = @. s*kdz/lambdas*imag(∂u.data*conj(u_saved))
    copyto!(∂dn, sum(g; dims = sdims))
end

compute_dn_gradient!(::Nothing, ::Nothing, ∂u, kdz, direction) = nothing

function backpropagate!(u::ScalarField, p::BPM, direction::Type{<:Direction};
        u_saved = nothing, ∂p = nothing)
    Nv = ndims(p.dn)
    n_slices = size(p.dn, Nv)
    dn_slices = eachslice(p.dn, dims = Nv)
    ∂dn_slices = isnothing(∂p) ?
                 Iterators.cycle(nothing) :
                 eachslice(∂p.dn, dims = Nv)
    u_saved_slices = isnothing(u_saved) ?
                     Iterators.cycle(nothing) :
                     eachslice(u_saved, dims = ndims(u_saved))
    propagate!(u, p.p_bpm_half, reverse(direction))
    for (dn, ∂dn, u_saved) in zip(
        @view(dn_slices[end:-1:2]),
        Iterators.reverse(∂dn_slices), Iterators.reverse(u_saved_slices))
        apply_dn_slice!(u, dn, p.kdz, reverse(direction))
        compute_dn_gradient!(∂dn, u_saved, u, p.kdz, direction)
        propagate!(u, p.p_bpm, reverse(direction))
    end
    apply_dn_slice!(u, dn_slices[1], p.kdz, reverse(direction))
    compute_dn_gradient!(∂dn_slices[1], u_saved_slices[1], u, p.kdz, direction)
    propagate!(u, p.p_bpm_half, reverse(direction))
    u
end

function backpropagate_with_gradient!(∂v::ScalarField, u_saved::AbstractArray,
        ∂p::NamedTuple, p::BPM{<:Trainable}, direction::Type{<:Direction})
    ∂u = backpropagate!(∂v, p, direction; u_saved, ∂p)
    (∂u, ∂p)
end
