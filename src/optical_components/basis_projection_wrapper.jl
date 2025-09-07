function compute_projection(r_basis, data)
    r_basis' * r_data
end

function set_projection!(r_data, r_basis, proj_coeffs)
    mul!(r_data, r_basis, proj_coeffs)
end

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
        set_projection!(r_mapped_data, r_basis, proj_coeffs)
        rD = typeof(r_mapped_data)
        ∂p = M == Trainable{Buffered} ? (; proj_coeffs = similar(proj_coeffs)) : nothing
        new{M, B, P, C, rD}(r_basis, proj_coeffs, wrapped_component, r_mapped_data, ∂p)
    end
end
