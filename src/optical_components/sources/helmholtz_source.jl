struct HelmholtzSource{M, Bf, Bb, H, Sf, Sb, P, T} <: AbstractCustomSource{M}
    trainability::Val{M}
    trainable_forward::Val{Bf}
    trainable_backward::Val{Bb}
    u0::H
    uf::H
    u0_fwd::Sf
    u0_bwd::Sb
    ∂p::P
    n0::T
end

Functors.@functor HelmholtzSource (u0_fwd, u0_bwd)

function HelmholtzSource(u::H;
                         n0::Real = 1.0,
                         trainable_forward::Bool = false,
                         trainable_backward::Bool = false,
                         buffered::Bool = false) where {U <: AbstractArray{<:Complex},
                                                        H <: HelmholtzField{U}}
    all_or_none_trainable = !xor(trainable_forward, trainable_backward)
    only_forward = trainable_forward && !trainable_backward
    only_backward = trainable_backward && !trainable_forward
    u0 = all_or_none_trainable ? copy(u) : nothing
    uf = all_or_none_trainable ? similar(u) : nothing
    M = trainability(trainable_forward || trainable_backward, buffered)
    u0_fwd, u0_bwd = all_or_none_trainable ? (nothing, nothing) : split_field(u; n0)
    ∂u0_fwd = only_forward ? similar(u0_fwd) : (;)
    ∂u0_bwd = only_backward ? similar(u0_bwd) : (;)
    ∂p = (trainable_forward
          && trainable_backward && buffered) ? (; u0 = similar(u0)) : nothing
    ∂p = ((only_forward || only_backward)
          && buffered) ? (; u0_fwd = ∂u0_fwd, u0_bwd = ∂u0_bwd) : ∂p
    HelmholtzSource(Val(M), Val(trainable_forward), Val(trainable_backward),
                    u0, uf, u0_fwd, u0_bwd, ∂p, n0)
end

Base.size(p::HelmholtzSource) = size(p.u0)
Base.size(p::HelmholtzSource, k::Integer) = size(p.u0, k)

trainable(p::HelmholtzSource{<:Trainable, true, false}) = (; u0_fwd = p.u0_fwd, u0_bwd = (;))
trainable(p::HelmholtzSource{<:Trainable, false, true}) = (; u0_fwd = (;), u0_bwd = p.u0_bwd)
trainable(p::HelmholtzSource{<:Trainable, true, true}) = (; u0 = p.u0)

get_preallocated_gradient(p::HelmholtzSource{Trainable{Buffered}}) = p.∂p

function propagate(p::HelmholtzSource{Static})
    copyto!(p.uf, p.u0)
    p.uf
end

function propagate(p::HelmholtzSource{<:Trainable, true, true})
    copyto!(p.uf, p.u0)
    p.uf
end

function propagate(p::HelmholtzSource{<:Trainable})
    HelmholtzField(p.u0_fwd, p.u0_bwd; n0 = p.n0)
end

propagate_and_save(p::HelmholtzSource) = propagate(p)

function backpropagate_with_gradient(∂v, ∂p::NamedTuple,
                                     p::HelmholtzSource{<:Trainable, Bf, Bb}) where {Bf, Bb}
    ∂v_fwd, ∂v_bwd = split_field(∂v)
    if Bf
        copyto!(∂p.u0_fwd, ∂v_fwd)
    end
    if Bb
        copyto!(∂p.u0_bwd, ∂v_bwd)
    end
    ∂p
end

function backpropagate_with_gradient(∂v, ∂p::NamedTuple,
                                     p::HelmholtzSource{<:Trainable, true, true})
    copyto!(∂p.u0, ∂v)
    ∂p
end

function Base.fill!(p::HelmholtzSource{Static}, u0::HelmholtzField)
    copyto!(p.u0, u0)
end

function Base.fill!(p::HelmholtzSource{<:Trainable, true, true}, u0::HelmholtzField)
    copyto!(p.u0, u0)
end

function Base.fill!(p::HelmholtzSource{<:Trainable}, u0::HelmholtzField)
    u0_fwd, u0_bwd = split_field(u0; n0 = p.n0)
    copyto!(p.u0_fwd, u0_fwd)
    copyto!(p.u0_bwd, u0_bwd)
    (u0_fwd, u0_bwd)
end

function get_source(p::HelmholtzSource{Static})
    p.u0
end

function get_source(p::HelmholtzSource{<:Trainable, true, true})
    p.u0
end

function get_source(p::HelmholtzSource{<:Trainable})
    (p.u0_fwd, p.u0_bwd)
end
