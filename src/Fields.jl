module Fields

export ScalarField
export get_data

import Base: +, -, *, /

struct ScalarField{U, T}
    data::U
    lambdas::T

    function ScalarField(
            u::U, lambdas::T
    ) where {N, U <: AbstractArray{<:Complex, N},
            T <: AbstractArray{<:Real}}
        @assert N >= 2
        nx, ny = size(u)
        nr = div(length(u), nx*ny)
        @assert length(lambdas) == nr
        lambdas = reshape(lambdas, (1, 1, size(u)[3:end]...))
        new{U, typeof(lambdas)}(u, lambdas)
    end

    function ScalarField(u::U, lambda::Real) where {U <: AbstractArray{<:Complex}}
        # ScalarField(u, [lambda])
        V = real(eltype(u))
        new{U, V}(u, V(lambda))
    end
end

function Base.broadcastable(sf::ScalarField)
    return Ref(sf)
end

function Base.broadcasted(f, a::ScalarField, b::AbstractArray)
    ScalarField(broadcast(f, a.data, b), a.lambdas)
end

function +(a::ScalarField, b::ScalarField)
    ScalarField(a.data + b.data, a.lambdas)
end

function +(a::ScalarField, b::AbstractArray)
    ScalarField(a.data + b, a.lambdas)
end

function -(a::ScalarField, b::ScalarField)
    ScalarField(a.data - b.data, a.lambdas)
end

function -(a::ScalarField, b::AbstractArray)
    ScalarField(a.data - b, a.lambdas)
end

Base.getindex(u::ScalarField, i...) = u.data[i...]
Base.size(u::ScalarField) = size(u.data)

function Base.copy(u::ScalarField)
    ScalarField(copy(u.data), u.lambdas)
end

function get_data(u::ScalarField)
    u.data
end

function get_data(u::AbstractArray)
    u
end

end
