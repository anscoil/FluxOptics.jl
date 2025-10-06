"""
    BasisProjectionWrapper(component, basis, coefficients)

Wrap a component to optimize over basis coefficients instead of pixel values.

Reparametrizes trainable components (Phase, Mask, etc.) to use a lower-dimensional
basis representation. Instead of optimizing all pixels independently, optimizes
coefficients that reconstruct the component via the basis. Useful for regularizing
ill-posed inverse problems or enforcing smoothness constraints.

# Arguments
- `component`: Trainable component to wrap (Phase, Mask, FourierPhase, etc.)
- `basis`: Basis functions for reconstruction (from `make_spatial_basis` or `make_fourier_basis`)
- `coefficients`: Initial coefficient array (length = number of basis functions)

# Use Case

Pixel-wise optimization can be ill-posed for inverse problems, leading to artifacts.
Basis projection restricts the solution space to smooth/structured functions,
providing implicit regularization.

# Examples
```julia
u = ScalarField(ones(ComplexF64, 256), (2.0,), 1.064)

# Phase mask with polynomial basis (instead of pixel-wise)
phase_mask = FourierPhase(u, zeros(256); trainable=true)

# Create polynomial basis
n_basis = 10
f_cutoff = 1e-3
basis = make_fourier_basis((fx, p) -> (abs(fx) < f_cutoff ? fx^p : 0.0), 
                            (256,), (2.0,), (0:n_basis-1))

# Wrap for basis optimization
phase_wrapper = BasisProjectionWrapper(phase_mask, basis, zeros(n_basis))

# Now optimize n_basis coefficients instead of nx pixels
system = source |> phase_wrapper |> propagator
# ... optimization optimizes basis coefficients ...
```

**Benefit:** Optimization converges with fewer parameters and enforces smoothness.

See also: [`make_spatial_basis`](@ref), [`make_fourier_basis`](@ref)
"""
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
    @assert Nd in (1, 2)
    r_args = map(x -> reshape(x, ntuple(k -> k <= Nd ? 1 : size(x, k-Nd), Nd+ndims(x))),
                 args)
    if Nd == 2
        xs = xs[1], xs[2]'
    end
    f.(xs..., r_args...)
end

"""
    make_spatial_basis(f, size, ds, params...)

Create basis functions evaluated on a spatial grid.

Generates a set of basis functions by evaluating function `f` with different
parameters on the spatial grid. Used with `BasisProjectionWrapper` to reparametrize
components.

# Arguments
- `f`: Basis function `(x, y, ..., params...) -> value`
- `size`: Grid size tuple
- `ds`: Spatial sampling tuple
- `params...`: Additional parameter arrays to broadcast (e.g., basis indices)

# Returns
Basis array where each slice corresponds to one basis function.

# Examples
```julia
# Polynomial basis in 1D
basis_1d = make_spatial_basis((x, n) -> x^n, (256,), (1.0,), 0:9)
# Creates 10 basis functions: x^0, x^1, ..., x^9

# Zernike-like radial polynomials in 2D
function zernike_radial(x, y, n, m)
    r = sqrt(x^2 + y^2)
    r < 1.0 ? r^n * cos(m*atan(y, x)) : 0.0
end

ns = [0, 1, 1, 2, 2, 2]
ms = [0, -1, 1, -2, 0, 2]
basis_2d = make_spatial_basis(zernike_radial, (128, 128), (1.0, 1.0), ns, ms)
```

See also: [`make_fourier_basis`](@ref), [`BasisProjectionWrapper`](@ref)
"""
function make_spatial_basis(f,
                            ns::NTuple{Nd, Integer},
                            ds::NTuple{Nd, Real},
                            args...) where {Nd}
    @assert Nd in (1, 2)
    make_basis(f, spatial_vectors(ns, ds), args...)
end

"""
    make_fourier_basis(f, size, ds, params...)

Create basis functions evaluated on a Fourier grid.

Similar to `make_spatial_basis` but evaluates on spatial frequency coordinates.
Useful for frequency-domain components like `FourierPhase`.

# Arguments
- `f`: Basis function `(fx, fy, ..., params...) -> value` in Fourier space
- `size`: Grid size tuple
- `ds`: Spatial sampling tuple (determines frequency grid)
- `params...`: Additional parameter arrays to broadcast

# Returns
Fourier-domain basis array.

# Examples
```julia
# Polynomial basis in Fourier domain
basis_fourier = make_fourier_basis((fx, n) -> abs(fx)^n, (256,), (1.0,), 0:5)

# Radial polynomial with cutoff
function radial_poly(fx, fy, p, NA, λ)
    f_mag = sqrt(fx^2 + fy^2)
    f_mag <= NA/λ ? f_mag^p : 0.0
end

NA = 0.01
λ = 1.064
powers = 0:2:10
basis = make_fourier_basis((fx, fy, p) -> radial_poly(fx, fy, p, NA, λ), (128, 128), (1.0, 1.0), powers)
```

See also: [`make_spatial_basis`](@ref), [`BasisProjectionWrapper`](@ref)
"""
function make_fourier_basis(f,
                            ns::NTuple{Nd, Integer},
                            ds::NTuple{Nd, Real},
                            args...) where {Nd}
    @assert Nd in (1, 2)
    fs = Tuple([fftfreq(nx, 1/dx) for (nx, dx) in zip(ns, ds)])
    make_basis(f, fs, args...)
end
