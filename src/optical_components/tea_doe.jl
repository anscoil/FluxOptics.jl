struct TeaDOE{M, Fn, Fr, A, U} <: AbstractCustomComponent{M}
    dn::Fn
    r::Fr
    h::A
    ∂p::Union{Nothing, @NamedTuple{h::A}}
    u::Union{Nothing, U}

    function TeaDOE(dn::Fn,
                    r::Fr,
                    h::A,
                    ∂p::Union{Nothing, @NamedTuple{h::A}},
                    u::U) where {Fn, Fr, A, U}
        M = isnothing(u) ? Trainable{Unbuffered} : Trainable{Buffered}
        new{M, Fn, Fr, A, U}(dn, r, h, ∂p, u)
    end

    function TeaDOE(u::ScalarField{U, Nd},
                    ds::NTuple{Nd, Real},
                    dn::Union{Real, Function},
                    f::Function = (_...) -> 0;
                    r::Union{Number, Function} = 1,
                    trainable::Bool = false,
                    buffered::Bool = false) where {N, Nd, T,
                                                   U <: AbstractArray{Complex{T}, N}}
        @assert Nd in (1, 2)
        @assert N >= Nd
        M = trainability(trainable, buffered)
        P = similar(U, real, Nd)
        ns = size(u)[1:Nd]
        h = P(function_to_array(f, ns, ds))
        ∂p = (trainable && buffered) ? (; h = similar(h)) : nothing
        u = (trainable && buffered) ? similar(u.electric) : nothing
        dn_f = isa(dn, Real) ? (λ -> T(dn)) : (λ -> T(dn(λ)))
        r_f = isa(r, Number) ? (λ -> Complex{T}(r)) : (λ -> Complex{T}(r(λ)))
        Fn = typeof(dn_f)
        Fr = typeof(r_f)
        A = typeof(h)
        new{M, Fn, Fr, A, U}(dn_f, r_f, h, ∂p, u)
    end

    function TeaDOE(u::ScalarField{U, Nd},
                    dn::Union{Real, Function},
                    f::Function = (_...) -> 0;
                    r::Union{Number, Function} = 1,
                    trainable::Bool = false,
                    buffered::Bool = false) where {U <: AbstractArray{<:Complex}, Nd}
        TeaDOE(u, Tuple(u.ds), dn, f; r, trainable, buffered)
    end
end

function TeaReflector(u::ScalarField{U, Nd},
                      ds::NTuple{Nd, Real},
                      f::Function = (_...) -> 0;
                      r::Union{Number, Function} = 1,
                      trainable::Bool = false,
                      buffered::Bool = false) where {U <: AbstractArray{<:Complex}, Nd}
    TeaDOE(u, ds, 2, f; r, trainable, buffered)
end

function TeaReflector(u::ScalarField{U, Nd},
                      f::Function = (_...) -> 0;
                      r::Union{Number, Function} = 1,
                      trainable::Bool = false,
                      buffered::Bool = false) where {U <: AbstractArray{<:Complex}, Nd}
    TeaDOE(u, 2, f; r, trainable, buffered)
end

Functors.@functor TeaDOE (h,)

get_data(p::TeaDOE) = p.h

trainable(p::TeaDOE{<:Trainable}) = (; h = p.h)

get_preallocated_gradient(p::TeaDOE{Trainable{Buffered}}) = p.∂p

alloc_saved_buffer(u::ScalarField, p::TeaDOE{Trainable{Unbuffered}}) = similar(u.electric)

get_saved_buffer(p::TeaDOE{Trainable{Buffered}}) = p.u

function apply_phase!(u::AbstractArray{T}, lambdas, p::TeaDOE, ::Type{Forward}) where {T}
    @. u *= p.r(lambdas) * cis((T(2)*π/lambdas)*p.dn(lambdas)*p.h)
end

function apply_phase!(u::AbstractArray{T}, lambdas, p::TeaDOE, ::Type{Backward}) where {T}
    @. u *= conj(p.r(lambdas)) * cis(-(T(2)*π/lambdas)*p.dn(lambdas)*p.h)
end

function propagate!(u::ScalarField, p::TeaDOE, direction::Type{<:Direction})
    apply_phase!(u.electric, get_lambdas(u), p, direction)
    u
end

function backpropagate!(u::ScalarField, p::TeaDOE, direction::Type{<:Direction})
    propagate!(u, p, reverse(direction))
end

function propagate_and_save!(u::ScalarField,
                             p::TeaDOE{Trainable{Buffered}},
                             direction::Type{<:Direction})
    copyto!(p.u, u.electric)
    propagate!(u, p, direction)
end

function propagate_and_save!(u::ScalarField,
                             u_saved,
                             p::TeaDOE{Trainable{Unbuffered}},
                             direction::Type{<:Direction})
    copyto!(u_saved, u.electric)
    propagate!(u, p, direction)
end

function compute_surface_gradient!(∂h::P,
                                   u_saved,
                                   ∂u::ScalarField,
                                   dn,
                                   r,
                                   direction) where {T <: Real, Nd,
                                                     P <: AbstractArray{T, Nd}}
    sdims = (Nd + 1):ndims(∂u)
    s = sign(direction)
    lambdas = get_lambdas(∂u)
    g = @. (s*T(2)*π*dn(lambdas)/lambdas)*imag(∂u.electric*conj(u_saved))
    copyto!(∂h, sum(g; dims = sdims))
end

function backpropagate_with_gradient!(∂v::ScalarField,
                                      u_saved::AbstractArray,
                                      ∂p::NamedTuple,
                                      p::TeaDOE{<:Trainable},
                                      direction::Type{<:Direction})
    ∂u = backpropagate!(∂v, p, direction)
    compute_surface_gradient!(∂p.h, u_saved, ∂u, p.dn, p.r, direction)
    (∂u, ∂p)
end
