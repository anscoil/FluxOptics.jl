struct FourierSmoothingWrapper{M, C, D, B, F, P} <: AbstractPureComponent{M}
    wrapped_component::C
    mapped_data::D
    buffer::B
    filter::F
    p_f::P
    ∂p::Union{Nothing, @NamedTuple{buffer::B}}
    
    function FourierSmoothingWrapper(wrapped_component::C,
                                     mapped_data::D,
                                     buffer::B,
                                     filter::F,
                                     p_f::P,
                                     ∂p::Union{Nothing, @NamedTuple{buffer::B}}) where {C, D, B, F, P}
        M = isnothing(∂p) ? Trainable{Unbuffered} : Trainable{Buffered}
        new{M, C, D, B, F, P}(wrapped_component, mapped_data, buffer, filter, p_f, ∂p)
    end

    function FourierSmoothingWrapper(wrapped_component::C,
                                     ns::NTuple{Nd, Integer},
                                     ds::NTuple{Nd, Real},
                                     f::Function) where {M <: Trainability,
                                                         C <: AbstractPipeComponent{M},
                                                         Nd}
        mapped_data = get_data(wrapped_component)
        @assert size(mapped_data)[1:Nd] == ns "Spatial dimensions $(size(data)[1:Nd]) don't match ns=$ns"
        D = typeof(mapped_data)
        F = similar(D, complex, Nd)
        filter = F(function_to_array(f, ns, ds, true) ./ prod(ns))
        buffer = similar(mapped_data, complex(eltype(mapped_data)))
        copyto!(buffer, mapped_data)
        B = typeof(buffer)
        p_f = make_fft_plans(buffer, Tuple(1:Nd); normalize=false)
        P = typeof(p_f)
        ∂p = M == Trainable{Buffered} ? (; buffer = similar(buffer)) : nothing
        new{M, C, D, B, F, P}(wrapped_component, mapped_data, buffer, filter, p_f, ∂p)
    end
end

Functors.@functor FourierSmoothingWrapper (buffer,)

get_data(p::FourierSmoothingWrapper) = p.buffer

trainable(p::FourierSmoothingWrapper{<:Trainable}) = (; buffer = p.buffer)

function apply_smoothing!(p::FourierSmoothingWrapper)
    p.p_f.ft * p.buffer
    p.buffer .*= p.filter
    p.p_f.ift * p.buffer
    copyto!(p.mapped_data, real.(p.buffer))
    p.wrapped_component
end

function propagate!(u::ScalarField, p::FourierSmoothingWrapper)
    wrapped_component = apply_smoothing!(p)
    propagate!(u, wrapped_component)
end

propagate(u::ScalarField, p::FourierSmoothingWrapper) = propagate!(copy(u), p)
