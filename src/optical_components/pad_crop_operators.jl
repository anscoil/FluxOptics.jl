function pad!(u_pad::AbstractArray{T, N}, u::AbstractArray{T, N};
              offset::NTuple{Nd, Integer} = ntuple(_ -> 0, Nd),
              pad_val = 0) where {T, N, Nd}
    @assert N >= Nd
    @assert all((>=)(0), offset)
    ns = size(u_pad)[1:Nd]
    for (k, n) in enumerate(offset)
        @assert size(u, k) + n <= ns[k]
    end
    pad_axes = ntuple(k -> k <= Nd ?
                           ((1 + offset[k]):(size(u, k) + offset[k])) :
                           axes(u, k), N)
    u_pad .= pad_val
    @views copyto!(u_pad[pad_axes...], u)
    u_pad
end

function pad(u::AbstractArray{T, N}, ns::NTuple{Nd, Integer};
             offset::NTuple{Nd, Integer} = ntuple(_ -> 0, Nd),
             pad_val = 0) where {T, N, Nd}
    shape = ntuple(k -> k <= Nd ? ns[k] : size(u, k), N)
    u_pad = similar(u, shape)
    pad!(u_pad, u; offset, pad_val)
end

function crop!(u::AbstractArray{T, N}, ns::NTuple{Nd, Integer};
               offset::NTuple{Nd, Integer} = ntuple(_ -> 0, Nd)) where {T, N, Nd}
    @assert N >= Nd
    @assert all((>=)(0), offset)
    for (k, n) in enumerate(offset)
        @assert ns[k] + n <= size(u, k)
    end
    crop_axes = ntuple(k -> k <= Nd ?
                            ((1 + offset[k]):(ns[k] + offset[k])) :
                            axes(u, k), N)
    u_crop = view(u, crop_axes...)
end

function crop!(u_crop::AbstractArray{T, N}, u::AbstractArray{T, N};
               offset::NTuple{Nd, Integer} = ntuple(_ -> 0, Nd)) where {T, N, Nd}
    ns = size(u_crop)[1:Nd]
    copyto!(u_crop, crop!(u, ns; offset))
    u_crop
end

function crop(u::AbstractArray{T, N}, ns::NTuple{Nd, Integer};
              offset::NTuple{Nd, Integer} = ntuple(_ -> 0, Nd)) where {T, N, Nd}
    shape = ntuple(k -> k <= Nd ? ns[k] : size(u, k), N)
    u_crop = similar(u, shape)
    copyto!(u_crop, crop!(u, ns; offset))
    u_crop
end

struct PadOperator{U, Nd, T, N}
    u_tmp::Union{Nothing, U}
    size_in::NTuple{N, Int}
    size_out::NTuple{N, Int}
    size_pad::NTuple{Nd, Int}
    offset::NTuple{Nd, Int}
    pad_value::T

    function PadOperator(u::U, ns::NTuple{Nd, Integer};
                         offset::NTuple{Nd, Integer} = ntuple(_ -> 0, Nd),
                         pad_val = 0) where {T, N, U <: AbstractArray{T, N}, Nd}
        size_in = size(u)
        size_out = ntuple(k -> k <= Nd ? ns[k] : size(u, k), N)
        new{U, Nd, T, N}(nothing, size_in, size_out, ns, offset, T(pad_val))
    end

    function PadOperator(u::U, u_tmp::U, nd::Integer;
                         offset::NTuple{Nd, Integer} = ntuple(_ -> 0, nd),
                         pad_val = 0) where {T, N, U <: AbstractArray{T, N}, Nd}
        @assert Nd == nd
        ns = size(u_tmp)[1:Nd]
        size_in = size(u)
        size_out = ntuple(k -> k <= Nd ? ns[k] : size(u, k), N)
        new{U, Nd, T, N}(u_tmp, size_in, size_out, ns, offset, T(pad_val))
    end

    function PadOperator(u::ScalarField{U, Nd}, ns::NTuple{Nd, Integer};
                         offset::NTuple{Nd, Integer} = ntuple(_ -> 0, Nd),
                         pad_val = 0) where {U, Nd}
        PadOperator(u.electric, ns; offset, pad_val)
    end

    function PadOperator(u::ScalarField{U, Nd}, u_tmp::U;
                         offset::NTuple{Nd, Integer} = ntuple(_ -> 0, Nd),
                         pad_val = 0) where {U, Nd}
        PadOperator(u.electric, u_tmp, Nd; offset, pad_val)
    end
end

function (p::PadOperator{U, Nd})(u::ScalarField{U, Nd}) where {U, Nd}
    electric_pad = if isnothing(p.u_tmp)
        pad(u.electric, p.size_pad; offset = p.offset, pad_val = p.pad_value)
    else
        pad!(p.u_tmp, u.electric; offset = p.offset, pad_val = p.pad_value)
    end
    set_field_data(u, electric_pad)
end

struct CropOperator{U, Nd, N}
    u_tmp::Union{Nothing, U}
    size_in::NTuple{N, Int}
    size_out::NTuple{N, Int}
    size_crop::NTuple{Nd, Int}
    offset::NTuple{Nd, Int}

    function CropOperator(u::U, ns::NTuple{Nd, Integer};
                          offset::NTuple{Nd, Integer} = ntuple(_ -> 0, Nd)) where {U, Nd}
        size_in = size(u)
        size_out = ntuple(k -> k <= Nd ? ns[k] : size(u, k), ndims(u))
        new{U, Nd, ndims(u)}(nothing, size_in, size_out, ns, offset)
    end

    function CropOperator(u::U, u_tmp::U, nd::Integer;
                          offset::NTuple{Nd, Integer} = ntuple(_ -> 0, nd)) where {U, Nd}
        @assert Nd == nd
        ns = size(u_tmp)[1:Nd]
        size_in = size(u)
        size_out = ntuple(k -> k <= Nd ? ns[k] : size(u, k), ndims(u))
        new{U, Nd, ndims(u)}(u_tmp, size_in, size_out, ns, offset)
    end

    function CropOperator(u::ScalarField{U, Nd}, ns::NTuple{Nd, Integer};
                          offset::NTuple{Nd, Integer} = ntuple(_ -> 0, Nd)) where {U, Nd}
        CropOperator(u.electric, ns; offset)
    end

    function CropOperator(u::ScalarField{U, Nd}, u_tmp::U;
                          offset::NTuple{Nd, Integer} = ntuple(_ -> 0, Nd)) where {U, Nd}
        CropOperator(u.electric, u_tmp, Nd; offset)
    end
end

function (p::CropOperator{U, Nd})(u::ScalarField{U, Nd}) where {U, Nd}
    electric_crop = if isnothing(p.u_tmp)
        crop(u.electric, p.size_crop; offset = p.offset)
    else
        crop!(p.u_tmp, u.electric; offset = p.offset)
    end
    set_field_data(u, electric_crop)
end

function (p::CropOperator{U, Nd})(u::ScalarField{U, Nd}, u_tmp::U) where {U, Nd}
    crop!(u_tmp, u.electric; offset = p.offset)
    set_field_data(u, u_tmp)
end

struct PadCropOperator{M, U, Nd, T, N} <: AbstractCustomComponent{M}
    p_pad::PadOperator{U, Nd, T, N}
    p_crop::CropOperator{U, Nd, N}
    u_tmp::Ref{U}
    ispad::Bool

    function PadCropOperator(p_pad::PadOperator{U, Nd, T, N},
                             p_crop::CropOperator{U, Nd, N},
                             u_tmp::Ref{U},
                             ispad::Bool) where {U, Nd, T, N}
        new{Static, U, Nd, T, N}(p_pad, p_crop, u_tmp, ispad)
    end

    function PadCropOperator(u::ScalarField{U, Nd}, u_tmp::U;
                             offset::NTuple{Nd, Integer} = ntuple(_ -> 0, Nd),
                             pad_val = 0) where {T, N, U <: AbstractArray{T, N}, Nd}
        p_pad = PadOperator(u, u_tmp; offset, pad_val)
        p_crop = CropOperator(u_tmp, size(u)[1:Nd])
        new{Static, U, Nd, T, N}(p_pad, p_crop, Ref{U}(), true)
    end
end

function Base.adjoint(p::PadCropOperator{M, U}) where {M, U}
    PadCropOperator(p.p_pad, p.p_crop, p.u_tmp, !p.ispad)
end

Functors.@functor PadCropOperator ()

function propagate!(u::ScalarField{U, Nd}, p::PadCropOperator{M, U, Nd},
                    ::Type{Forward}) where {M, U, Nd}
    if p.ispad
        p.u_tmp[] = u.electric
        p.p_pad(u)
    else
        p.p_crop(u, p.u_tmp[])
    end
end

function propagate!(u::ScalarField{U, Nd}, p::PadCropOperator{M, U, Nd},
                    ::Type{Backward}) where {M, U, Nd}
    if !p.ispad
        p.u_tmp[] = u.electric
        p.p_pad(u)
    else
        p.p_crop(u, p.u_tmp[])
    end
end

function backpropagate!(∂v::ScalarField, p::PadCropOperator, direction::Type{<:Direction})
    propagate!(∂v, p, reverse(direction))
end
