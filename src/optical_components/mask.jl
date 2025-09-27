struct Mask{M, A, U} <: AbstractCustomComponent{M}
    m::A
    ∂p::Union{Nothing, @NamedTuple{m::A}}
    u::Union{Nothing, U}

    function Mask(m::A, ∂p::Union{Nothing, @NamedTuple{m::A}}, u::U) where {A, U}
        M = isnothing(u) ? Trainable{Unbuffered} : Trainable{Buffered}
        new{M, A, U}(m, ∂p, u)
    end

    function Mask(u::ScalarField{U, Nd},
                  ds::NTuple{Nd, Real},
                  f::Union{Function, AbstractArray} = (_...) -> 1;
                  trainable::Bool = false,
                  buffered::Bool = false) where {Nd, U}
        M = trainability(trainable, buffered)
        @assert Nd in (1, 2)
        if isa(f, Function)
            A = similar(U, Nd)
            ns = size(u)[1:Nd]
            m = A(function_to_array(f, ns, ds))
        else
            @assert isbroadcastable(f, u)
            A = similar(U, ndims(f))
            m = A(f)
        end
        ∂p = (trainable && buffered) ? (; m = similar(m)) : nothing
        u = (trainable && buffered) ? similar(u.electric) : nothing
        A = typeof(m)
        new{M, A, U}(m, ∂p, u)
    end

    function Mask(u::ScalarField{U, Nd},
                  f::Union{Function, AbstractArray} = (_...) -> 1;
                  trainable::Bool = false,
                  buffered::Bool = false) where {Nd, U}
        Mask(u, u.ds, f; trainable, buffered)
    end
end

Functors.@functor Mask (m,)

get_data(p::Mask) = p.m

trainable(p::Mask{<:Trainable}) = (; m = p.m)

get_preallocated_gradient(p::Mask{Trainable{Buffered}}) = p.∂p

alloc_saved_buffer(u::ScalarField, p::Mask{Trainable{Unbuffered}}) = similar(u.electric)

get_saved_buffer(p::Mask{Trainable{Buffered}}) = p.u

function propagate!(u::ScalarField,
                    p::Mask,
                    direction::Type{<:Direction};
                    u_saved = nothing)
    copyto!(u_saved, u.electric)
    @. u.electric *= conj_direction(p.m, direction)
    u
end

function propagate_and_save!(u::ScalarField,
                             p::Mask{Trainable{Buffered}},
                             direction::Type{<:Direction})
    propagate!(u, p, direction; u_saved = p.u)
end

function propagate_and_save!(u::ScalarField,
                             u_saved::AbstractArray,
                             p::Mask{Trainable{Unbuffered}},
                             direction::Type{<:Direction})
    propagate!(u, p, direction; u_saved)
end

function compute_mask_gradient!(∂m::AbstractArray{<:Complex, Nd},
                                u_saved,
                                ∂u::ScalarField,
                                direction) where {Nd}
    sdims = (Nd + 1):ndims(∂u.electric)
    g = @. conj_direction(∂u.electric*conj(u_saved), direction)
    copyto!(∂m, sum(g; dims = sdims))
end

compute_mask_gradient!(::Nothing, ::Nothing, ∂u, direction) = nothing

function backpropagate!(u::ScalarField,
                        p::Mask,
                        direction::Type{<:Direction};
                        u_saved = nothing,
                        ∂p = nothing)
    ∂m = isnothing(∂p) ? nothing : ∂p.m
    compute_mask_gradient!(∂m, u_saved, u, direction)
    @. u.electric *= conj_direction(p.m, reverse(direction))
    u
end

function backpropagate_with_gradient!(∂v::ScalarField,
                                      u_saved::AbstractArray,
                                      ∂p::NamedTuple,
                                      p::Mask{<:Trainable},
                                      direction::Type{<:Direction})
    ∂u = backpropagate!(∂v, p, direction; u_saved, ∂p)
    (∂u, ∂p)
end
