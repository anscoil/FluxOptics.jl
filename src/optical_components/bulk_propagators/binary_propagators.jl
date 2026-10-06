abstract type AbstractBinaryPropagator{M} <: AbstractBidirectionalComponent{M} end

# Subtypes provide the fields mask_xyz, mask_eps, eps_1, eps_2, dz, medium_1, medium_2,
# factors_1, factors_2, u_tmp, p_f and nrm_f.

function binary_maps(a::AbstractArray, mask_xyz, mask_eps, eps_1, eps_2)
    T = real(eltype(a))
    mask = similar(a, T, size(mask_xyz))
    copyto!(mask, mask_xyz)
    eps_xyz = similar(a, Complex{T}, size(mask_xyz))
    if isnothing(mask_eps)
        @. eps_xyz = eps_1 * mask + eps_2 * (1 - mask)
    else
        copyto!(eps_xyz, mask_eps)
    end
    mask, eps_xyz
end

component_arrays(u::AbstractField) = Tuple(StructArrays.components(StructArray(u)))

binary_coefs(p::AbstractBinaryPropagator) =
    (p.medium_1.modes, p.factors_1, p.medium_2.modes, p.factors_2)

function split_media(ψ, m, ε, eps_1, eps_2, c, nrm)
    kick(ψ, m * nrm, ε, eps_1, c), kick(ψ, (1 - m) * nrm, ε, eps_2, c)
end

function split_media_adjoint(∂ψ1, ∂ψ2, m, ε, eps_1, eps_2, c, nrm)
    kick_adjoint(∂ψ1, m * nrm, ε, eps_1, c) +
        kick_adjoint(∂ψ2, (1 - m) * nrm, ε, eps_2, c)
end

function propagate_binary(m1, f1, m2, f2, direction::Direction, ψ1::Tuple, ψ2::Tuple)
    Ψ1, _ = propagate_modes(m1, f1, direction, ψ1, ())
    Ψ2, _ = propagate_modes(m2, f2, direction, ψ2, ())
    map(+, Ψ1, Ψ2)
end

function propagate_binary_adjoint(m1, f1, m2, f2, direction::Direction,
                                  ∂ψ::Tuple, ::Tuple)
    ∂ψ1, _ = propagate_modes_adjoint(m1, f1, direction, ∂ψ, ())
    ∂ψ2, _ = propagate_modes_adjoint(m2, f2, direction, ∂ψ, ())
    ∂ψ1, ∂ψ2
end

function split_slice!(u::AbstractField, v::AbstractField, p::AbstractBinaryPropagator,
                      k::Integer, c)
    ψu, ψv = StructArray(u), StructArray(v)
    mask_k = view(p.mask_xyz, :, :, k)
    eps_k = view(p.mask_eps, :, :, k)
    StructArray((ψu, ψv)) .=
        split_media.(ψu, mask_k, eps_k, p.eps_1, p.eps_2, c, p.nrm_f)
end

function split_slice_adjoint!(∂u::AbstractField, v::AbstractField,
                              p::AbstractBinaryPropagator, k::Integer, c)
    ∂ψu, ∂ψv = StructArray(∂u), StructArray(v)
    mask_k = view(p.mask_xyz, :, :, k)
    eps_k = view(p.mask_eps, :, :, k)
    ∂ψu .= split_media_adjoint.(∂ψu, ∂ψv, mask_k, eps_k, p.eps_1, p.eps_2, c, p.nrm_f)
end

function merge_slice!(u::AbstractField, v::AbstractField, p::AbstractBinaryPropagator,
                      direction::Direction)
    launch_merge!(propagate_binary, component_arrays(u), component_arrays(v),
                  binary_coefs(p), direction)
end

function merge_slice_adjoint!(∂u::AbstractField, v::AbstractField,
                              p::AbstractBinaryPropagator, direction::Direction)
    launch_modal!(propagate_binary_adjoint, component_arrays(∂u), component_arrays(v),
                  binary_coefs(p), direction)
end

function propagate_slice!(u::AbstractField, p::AbstractBinaryPropagator, k::Integer,
                          direction::Direction)
    v = p.u_tmp
    c = kick_coefficient(u, p.dz, direction)
    compute_ift!(p.p_f, u)
    split_slice!(u, v, p, k, c)
    compute_ft!(p.p_f, u)
    compute_ft!(p.p_f, v)
    merge_slice!(u, v, p, direction)
    u
end

function propagate_slice_adjoint!(∂u::AbstractField, p::AbstractBinaryPropagator,
                                  k::Integer, direction::Direction)
    v = p.u_tmp
    c = kick_coefficient(∂u, p.dz, direction)
    merge_slice_adjoint!(∂u, v, p, direction)
    compute_ift!(p.p_f, ∂u)
    compute_ift!(p.p_f, v)
    split_slice_adjoint!(∂u, v, p, k, c)
    compute_ft!(p.p_f, ∂u)
    ∂u
end

function propagate!(u::AbstractField, state, activations, p::AbstractBinaryPropagator,
                    direction::Direction)
    for k in reverse(1:size(p.mask_xyz, 3), direction)
        propagate_slice!(u, p, k, direction)
    end
    u
end

function propagate_adjoint!(∂u::AbstractField, ::Nothing, state, ::Nothing,
                            p::AbstractBinaryPropagator, direction::Direction)
    for k in reverse(1:size(p.mask_xyz, 3), reverse(direction))
        propagate_slice_adjoint!(∂u, p, k, direction)
    end
    ∂u
end
