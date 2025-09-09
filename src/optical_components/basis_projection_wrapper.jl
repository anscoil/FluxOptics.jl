struct BasisProjectionWrapper{M, B, P, C, D} <: AbstractCustomComponent{M}
    basis::B
    proj_coeffs::P
    wrapped_component::C
    mapped_data::D
    ∂p::Union{Nothing, @NamedTuple{proj_coeffs::P}}

    function BasisProjectionWrapper(basis::B, proj_coeffs::P, wrapped_component::C,
            mapped_data::D, ∂p::Union{Nothing, @NamedTuple{proj_coeffs::P}}
    ) where {B, P, C, D}
        M = isnothing(∂p) ? Trainable{Unbuffered} : Trainable{Buffered}
        new{M, B, P, C, D}(basis, proj_coeffs, wrapped_component, mapped_data, ∂p)
    end

    function BasisProjectionWrapper(
            wrapped_component::C, basis::AbstractArray,
            proj_coeffs::AbstractArray
    ) where {M <: Trainability, C <: AbstractCustomComponent{M}}
        mapped_data = get_data(wrapped_component)
        D = typeof(mapped_data)
        mdims = ndims(mapped_data)
        bdims = ndims(basis)
        @assert bdims > mdims
        m_size = size(mapped_data)
        b_size = size(basis)
        @assert b_size[1:mdims] == m_size
        @assert size(proj_coeffs) == b_size[(mdims + 1):end]
        nd = length(mapped_data)
        r_mapped_data = reshape(mapped_data, nd)
        nb = length(basis)
        B = adapt_dim(D, 2)
        r_basis = B(reshape(basis, (nd, div(nb, nd))))
        nc = length(proj_coeffs)
        P = adapt_dim(D, 1)
        proj_coeffs = P(reshape(proj_coeffs, nc))
        rD = typeof(r_mapped_data)
        ∂p = M == Trainable{Buffered} ? (; proj_coeffs = similar(proj_coeffs)) : nothing
        new{M, B, P, C, rD}(r_basis, proj_coeffs, wrapped_component, r_mapped_data, ∂p)
    end
end

Functors.@functor BasisProjectionWrapper (proj_coeffs,)

get_data(p::BasisProjectionWrapper) = p.proj_coeffs

get_wrapped_data(p::BasisProjectionWrapper) = get_data(p.wrapped_component)

trainable(p::BasisProjectionWrapper{<:Trainable}) = (; proj_coeffs = p.proj_coeffs)

function alloc_gradient(p::BasisProjectionWrapper{Trainable{Unbuffered}})
    (map(similar, trainable(p)), alloc_gradient(p.wrapped_component))
end

function get_preallocated_gradient(p::BasisProjectionWrapper{Trainable{Buffered}})
    (p.∂p, get_preallocated_gradient(p.wrapped_component))
end

function alloc_saved_buffer(u::ScalarField,
        p::BasisProjectionWrapper{Trainable{Unbuffered}})
    alloc_saved_buffer(u, p.wrapped_component)
end

function get_saved_buffer(p::BasisProjectionWrapper{Trainable{Buffered}})
    get_saved_buffer(p.wrapped_component)
end

function set_basis_projection!(r_data, r_basis, proj_coeffs)
    mul!(r_data, r_basis, proj_coeffs)
end

function propagate!(u::ScalarField, p::BasisProjectionWrapper, direction::Type{<:Direction})
    set_basis_projection!(p.mapped_data, p.basis, p.proj_coeffs)
    propagate!(u, p.wrapped_component, direction)
end

function propagate_and_save!(
        u::ScalarField, p::BasisProjectionWrapper{Trainable{Buffered}},
        direction::Type{<:Direction})
    set_basis_projection!(p.mapped_data, p.basis, p.proj_coeffs)
    propagate_and_save!(u, p.wrapped_component, direction)
end

function propagate_and_save!(u::ScalarField, u_saved::AbstractArray,
        p::BasisProjectionWrapper{Trainable{Unbuffered}}, direction::Type{<:Direction})
    set_basis_projection!(p.mapped_data, p.basis, p.proj_coeffs)
    propagate_and_save!(u, u_saved, p.wrapped_component, direction)
end

function compute_basis_projection!(proj_coeffs, r_basis, r_data)
    mul!(proj_coeffs, r_basis', r_data)
end

function backpropagate!(u::ScalarField, p::BasisProjectionWrapper,
        direction::Type{<:Direction})
    backpropagate!(u, p.wrapped_component, direction)
end

function backpropagate_with_gradient!(∂v::ScalarField, u_saved::AbstractArray,
        ∂p::Tuple{NamedTuple, NamedTuple}, p::BasisProjectionWrapper{<:Trainable},
        direction::Type{<:Direction})
    ∂p_coeffs, ∂p_wrapped = ∂p
    (∂u,
        (∂mapped_data,)) = backpropagate_with_gradient!(∂v, u_saved, ∂p_wrapped,
        p.wrapped_component, direction)
    compute_basis_projection!(∂p_coeffs.proj_coeffs, p.basis, reshape(∂mapped_data, :))
    (∂u, ∂p_coeffs)
end

function make_basis(f, xs::NTuple{Nd, AbstractArray{<:Real}}, args...) where {Nd}
    r_args = map(
        x -> reshape(x, ntuple(k -> k <= Nd ? 1 : size(x, k-Nd), Nd+ndims(x))), args)
    f.(xs..., r_args...)
end

function make_spatial_basis(f, ns::NTuple{Nd, Integer}, ds::NTuple{Nd, Real},
        args...) where {Nd}
    @assert Nd in (1, 2)
    make_basis(f, spatial_vectors(ns, ds), args...)
end

function make_fourier_basis(f, ns::NTuple{Nd, Integer}, ds::NTuple{Nd, Real},
        args...) where {Nd}
    @assert Nd in (1, 2)
    fs = Tuple([fftfreq(nx, 1/dx) for (nx, dx) in zip(ns, ds)])
    make_basis(f, fs, args...)
end
