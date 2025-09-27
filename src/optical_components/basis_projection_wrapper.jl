struct BasisProjectionWrapper{M, B, P, C, D} <: AbstractPureComponent{M}
    basis::B
    proj_coeffs::P
    wrapped_component::C
    mapped_data::D
    ∂p::Union{Nothing, @NamedTuple{proj_coeffs::P}}

    function BasisProjectionWrapper(basis::B,
                                    proj_coeffs::P,
                                    wrapped_component::C,
                                    mapped_data::D,
                                    ∂p::Union{Nothing, @NamedTuple{proj_coeffs::P}}) where {B,
                                                                                            P,
                                                                                            C,
                                                                                            D}
        M = isnothing(∂p) ? Trainable{Unbuffered} : Trainable{Buffered}
        new{M, B, P, C, D}(basis, proj_coeffs, wrapped_component, mapped_data, ∂p)
    end

    function BasisProjectionWrapper(wrapped_component::C,
                                    basis::AbstractArray,
                                    proj_coeffs::AbstractArray) where {M <: Trainability,
                                                                       C <:
                                                                       AbstractPipeComponent{M}}
        mapped_data = get_data(wrapped_component)
        if !isa(mapped_data, AbstractArray)
            mapped_data = filter(x -> isa(x, AbstractArray), get_data(wrapped_component))
            if length(mapped_data) > 1
                @warn "Calling get_data on a Fourier wrapper with multiple components \
                   recovers only the data of the first component."
            end
            mapped_data = first(mapped_data)
        end
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
        B = similar(D, 2)
        r_basis = B(reshape(basis, (nd, div(nb, nd))))
        nc = length(proj_coeffs)
        P = similar(D, 1)
        proj_coeffs = P(reshape(proj_coeffs, nc))
        rD = typeof(r_mapped_data)
        ∂p = M == Trainable{Buffered} ? (; proj_coeffs = similar(proj_coeffs)) : nothing
        new{M, B, P, C, rD}(r_basis, proj_coeffs, wrapped_component, r_mapped_data, ∂p)
    end
end

Functors.@functor BasisProjectionWrapper (proj_coeffs,)

get_data(p::BasisProjectionWrapper) = p.proj_coeffs

trainable(p::BasisProjectionWrapper{<:Trainable}) = (; proj_coeffs = p.proj_coeffs)

function set_basis_projection!(p::BasisProjectionWrapper)
    mul!(p.mapped_data, p.basis, p.proj_coeffs)
    p.wrapped_component
end

function propagate(u::ScalarField, p::BasisProjectionWrapper, direction::Type{<:Direction})
    wrapped_component = set_basis_projection!(p)
    propagate!(u, wrapped_component, direction)
end

function make_basis(f, xs::NTuple{Nd, AbstractArray{<:Real}}, args...) where {Nd}
    r_args = map(x -> reshape(x, ntuple(k -> k <= Nd ? 1 : size(x, k-Nd), Nd+ndims(x))),
                 args)
    f.(xs..., r_args...)
end

function make_spatial_basis(f,
                            ns::NTuple{Nd, Integer},
                            ds::NTuple{Nd, Real},
                            args...) where {Nd}
    @assert Nd in (1, 2)
    make_basis(f, spatial_vectors(ns, ds), args...)
end

function make_fourier_basis(f,
                            ns::NTuple{Nd, Integer},
                            ds::NTuple{Nd, Real},
                            args...) where {Nd}
    @assert Nd in (1, 2)
    fs = Tuple([fftfreq(nx, 1/dx) for (nx, dx) in zip(ns, ds)])
    make_basis(f, fs, args...)
end
