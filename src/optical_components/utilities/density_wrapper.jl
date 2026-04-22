struct DensityWrapper{M, C, A, T} <: AbstractPureComponent{M}
    wrapped_component::C
    mapped_data::A
    D::A
    h::A
    β::Ref{T}
    offset::T
    ∂p::Union{Nothing, @NamedTuple{D::A, h::A}}

    function DensityWrapper(wrapped_component::C,
                            mapped_data::A,
                            D::A,
                            h::A,
                            β::Ref{T},
                            offset::T,
                            ∂p::Union{Nothing, @NamedTuple{D::A, h::A}}) where {C, A, T}
        M = isnothing(∂p) ? Trainable{Unbuffered} : Trainable{Buffered}
        new{M, C, A, T}(wrapped_component, mapped_data, D, h, β, offset, ∂p)
    end

    function DensityWrapper(wrapped_component::C,
                            ns::NTuple{Nd, Integer},
                            h::Real,
                            β::Real = 1.0,
                            offset::Real = -h/2) where {M <: Trainability,
                                                       C <: AbstractPipeComponent{M},
                                                       Nd}
        mapped_data = get_data(wrapped_component)
        T = eltype(mapped_data)
        @assert T <: Real "mapped_data must be real-valued, got $(eltype(mapped_data))"
        @assert size(mapped_data)[1:Nd] == ns "Spatial dimensions $(size(mapped_data)[1:Nd]) don't match ns=$ns"
        A = typeof(mapped_data)
        D = similar(mapped_data)
        fill!(D, 0.0)
        h_arr = similar(mapped_data, ntuple(_ -> 1, ndims(mapped_data)))
        fill!(h_arr, h)
        ∂p = M == Trainable{Buffered} ? (; D = similar(D), h = similar(h_arr)) : nothing
        new{M, C, A, T}(wrapped_component, mapped_data, D, h_arr, Ref{T}(β), T(offset), ∂p)
    end
end

Functors.@functor DensityWrapper (D, h)

get_data(p::DensityWrapper) = p.D

trainable(p::DensityWrapper{<:Trainable}) = (; D = p.D, h = p.h)

set_sharpness!(p::DensityWrapper, β::Real) = p.β[] = β

function sigmoid(x::T) where {T <: Real}
    T(1) / (T(1) + exp(-x))
end

function apply_projection!(p::DensityWrapper)
    @. p.mapped_data = p.offset + p.h * sigmoid(p.β[] * p.D)
    p.wrapped_component
end

function propagate!(u::ScalarField, p::DensityWrapper)
    wrapped_component = apply_projection!(p)
    propagate!(u, wrapped_component)
end

propagate(u::ScalarField, p::FourierSmoothingWrapper) = propagate!(copy(u), p)
