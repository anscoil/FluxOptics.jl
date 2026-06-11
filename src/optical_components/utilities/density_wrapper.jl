struct DensityWrapper{M, C, A, AD, T} <: AbstractPureComponent{M}
    wrapped_component::C
    mapped_data::A
    aux_data::AD
    D::A
    h::A
    β::Ref{T}
    offset::T
    binarize::Ref{Bool}
    ∂p::Union{Nothing, @NamedTuple{D::A, h::A}}

    function DensityWrapper(wrapped_component::C,
                            mapped_data::A,
                            aux_data::AD,
                            D::A,
                            h::A,
                            β::Ref{T},
                            offset::T,
                            binarize::Ref{Bool},
                            ∂p::Union{Nothing, @NamedTuple{D::A, h::A}}) where {C, A, AD, T}
        M = isnothing(∂p) ? Trainable{Unbuffered} : Trainable{Buffered}
        new{M, C, A, AD, T}(wrapped_component, mapped_data, aux_data, D, h, β, offset, binarize, ∂p)
    end

    function DensityWrapper(wrapped_component::C,
                            ns::NTuple{Nd, Integer},
                            h::Real;
                            sharpness::Real = 1.0,
                            offset::Real = 0,
                            binarize::Bool = false) where {M <: Trainability,
                                                           C <: AbstractPipeComponent{M}, Nd}
        mapped_data = get_data(wrapped_component)
        T = eltype(mapped_data)
        @assert T <: Real "mapped_data must be real-valued, got $(eltype(mapped_data))"
        @assert size(mapped_data)[1:Nd] == ns "Spatial dimensions $(size(mapped_data)[1:Nd]) don't match ns=$ns"
        A = typeof(mapped_data)
        D = similar(mapped_data)
        fill!(D, 0.0)
        h_arr = similar(mapped_data, ntuple(_ -> 1, ndims(mapped_data)))
        fill!(h_arr, h)
        aux_data = auxiliary_trainable(wrapped_component)
        AD = typeof(aux_data)
        ∂p = M == Trainable{Buffered} ? (; D = similar(D), h = similar(h_arr)) : nothing
        new{M, C, A, AD, T}(wrapped_component, mapped_data, aux_data,
                            D, h_arr, Ref{T}(sharpness), T(offset), Ref(binarize), ∂p)
    end
end

Functors.@functor DensityWrapper (D, h, aux_data)

data_symbol_chain(p::DensityWrapper) = (:D,)

function trainable(p::DensityWrapper{<:Trainable})
    (; D = p.D, h = p.h, aux_data = p.aux_data)
end

set_sharpness!(p::DensityWrapper, β::Real) = p.β[] = β

set_binarize!(p::DensityWrapper, binarize::Bool) = p.binarize[] = binarize

function sigmoid(x::T) where {T <: Real}
    T(1) / (T(1) + exp(-x))
end

function apply_projection!(p::DensityWrapper)
    @. p.mapped_data = p.offset + p.h * sigmoid(p.β[] * p.D)
    if p.binarize[]
        @. p.mapped_data = ifelse(p.mapped_data > (p.offset + p.h/2), p.offset + p.h, p.offset)
    end
    p.wrapped_component
end

function propagate!(u::ScalarField, p::DensityWrapper)
    wrapped_component = apply_projection!(p)
    propagate!(u, wrapped_component)
end

propagate(u::ScalarField, p::DensityWrapper) = propagate!(copy(u), p)
