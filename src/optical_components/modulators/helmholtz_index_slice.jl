struct HelmholtzIndexSlice{M, A, P, H, T} <: AbstractCustomComponent{M}
    trainability::Val{M}
    index_slice::A
    dz::T
    n0::T
    ∂p::P
    u::H
end

Functors.@functor HelmholtzIndexSlice (index_slice,)

function HelmholtzIndexSlice(u::H,
                             dz::Real,
                             n0::Real = 1.0,
                             f::Union{Function, AbstractArray{<:Number, 2}} = (x,y) -> n0;
                             trainable::Bool = false,
                             buffered::Bool = false) where {T, U <: AbstractArray{Complex{T}},
                                                            H <: HelmholtzField{U}}
    ns = size(u)[1:2]
    M = trainability(trainable, buffered)
    raw = if isa(f, Function)
        function_to_array(f, ns, u.ds)
    else
        @assert isbroadcastable(f, u)
        f
    end
    E = eltype(raw) <: Real ? T : Complex{T}
    index_slice = similar(u.electric, E, ns)
    copyto!(index_slice, raw)
    ∂p = (trainable && buffered) ? (; index_slice = similar(index_slice)) : nothing
    u = (trainable && buffered) ? similar(u.electric) : nothing
    HelmholtzIndexSlice(Val(M), index_slice, T(dz), T(n0), ∂p, u)
end

data_symbol(p::HelmholtzIndexSlice) = :index_slice

trainable(p::HelmholtzIndexSlice{<:Trainable}) = (; index_slice = p.index_slice)

get_preallocated_gradient(p::HelmholtzIndexSlice{Trainable{Buffered}}) = p.∂p

function alloc_saved_buffer(u::HelmholtzField,
                            p::HelmholtzIndexSlice{Trainable{Unbuffered}})
    similar(u.electric)
end

get_saved_buffer(p::HelmholtzIndexSlice{Trainable{Buffered}}) = p.u

function propagate!(u::HelmholtzField, p::HelmholtzIndexSlice)
    @. u.electric_dz += ((2π/u.lambdas.val)^2
                         * (p.n0^2 - p.index_slice^2) * p.dz
                         * u.electric)
    u
end

function propagate_and_save!(u::HelmholtzField,
                             u_saved::AbstractArray,
                             p::HelmholtzIndexSlice{<:Trainable})
    copyto!(u_saved, u.electric)
    propagate!(u, p)
end

function backpropagate!(u::HelmholtzField, p::HelmholtzIndexSlice)
    @. u.electric += ((2π/u.lambdas.val)^2
                      * conj(p.n0^2 - p.index_slice^2) * p.dz
                      * u.electric_dz)
    u
end

function backpropagate_with_gradient!(∂v::HelmholtzField,
                                      u_saved::AbstractArray,
                                      ∂p::NamedTuple,
                                      p::HelmholtzIndexSlice{<:Trainable})
    k0sq = (2π / ∂v.lambdas.val)^2
    _index_slice_grad!(∂p.index_slice, ∂v.electric_dz, u_saved, p.index_slice, k0sq, p.dz)
    backpropagate!(∂v, p)
end

function _index_slice_grad!(∂n::AbstractArray{<:Real}, ∂dEdz, E, n, k0sq, dz)
    @. ∂n += real(conj(∂dEdz) * E) * (-2 * k0sq * n * dz)
end

function _index_slice_grad!(∂n::AbstractArray{<:Complex}, ∂dEdz, E, n, k0sq, dz)
    @. ∂n += conj(∂dEdz) * E * (-2 * k0sq * conj(n) * dz)
end
