using Polynomials
using LinearAlgebra

function get_zR(w0, λ)
    π*w0^2/λ
end

function get_ψG(z, zR)
    atan(z, zR)
end

function get_w(w0, λ, z)
    zR = get_zR(w0, λ)
    w0*sqrt(1+(z/zR)^2)
end

function get_q(w0, λ, z)
    zR = get_zR(w0, λ)
    z - im*zR
end

function gaussian_normalization_constant(w0)
    sqrt(sqrt(2/π)/w0)
end

function hg_normalization_constant(m)
    1 / sqrt(2^m*factorial(m))
end

function hg_normalization_constant(w0, m)
    gaussian_normalization_constant(w0) * hg_normalization_constant(m)
end

function lg_normalization_constant(w0, p, l)
    sqrt(2*factorial(p) / (π*factorial(p + abs(l))))/w0
end

"""
    Gaussian1D(w0::Real; norm_constant=nothing)
    Gaussian1D(w0::Real, λ::Real, z::Real; constant_phase=true, norm_constant=nothing)

Create a one-dimensional Gaussian mode.

The first form creates a Gaussian at the beam waist, while the second includes propagation
effects at distance `z` from the waist.

# Arguments
- `w0::Real`: Beam waist radius
- `λ::Real`: Wavelength (for propagated version)
- `z::Real`: Propagation distance from waist (for propagated version)
- `constant_phase=true`: Include exp(ikz) phase factor
- `norm_constant=nothing`: Custom normalization constant - defines the peak intensity (default uses proper Gaussian normalization)

# Returns
`Gaussian1D` mode that can be evaluated at spatial positions.

# Examples
```jldoctest
julia> g = Gaussian1D(10.0);  # 10 μm waist at focus

julia> nx = 40;  # 40 points

julia> dx = 4.0;  # in µm

julia> x, = spatial_vectors(40, dx);

julia> amplitudes = g(x);

julia> sum(abs2, amplitudes) * dx  # Normalization to 1 by default
0.9999999999999194

julia> g_prop = Gaussian1D(10.0, 1.064, 1000.0);  # Propagated 1 mm

julia> amplitudes_prop = g_prop(x);

julia> sum(abs2, amplitudes_prop) * dx
0.9999943872148783
```

See also: [`HermiteGaussian1D`](@ref), [`Gaussian`](@ref), [`HermiteGaussian`](@ref), [`LaguerreGaussian`](@ref)
"""
struct Gaussian1D{T, P <: Union{Nothing, <:NamedTuple}} <: Mode{1, T}
    w0::T
    C::Complex{T}
    data::P

    function Gaussian1D(w0::Real,
                        λ::Real,
                        z::Real;
                        constant_phase = true,
                        norm_constant = nothing)
        T = float(eltype(w0))
        q0 = Complex{T}(get_q(w0, λ, 0))
        qz = Complex{T}(get_q(w0, λ, z))
        wz = Complex{T}(get_w(w0, λ, z))
        k = T(2π/λ)
        eikz = constant_phase ? Complex{T}(exp(im*k*z)) : Complex{T}(1)
        normalization_constant = isnothing(norm_constant) ?
                                 gaussian_normalization_constant(w0) :
                                 sqrt(norm_constant)
        C = normalization_constant * sqrt(q0 / qz)
        data = (; λ, z, wz, qz, eikz, e_arg = im*k/(2*qz))
        new{T, typeof(data)}(w0, C, data)
    end

    function Gaussian1D(w0::Real; norm_constant = nothing)
        T = float(eltype(w0))
        normalization_constant = isnothing(norm_constant) ?
                                 gaussian_normalization_constant(w0) :
                                 sqrt(norm_constant)
        C = Complex{T}(normalization_constant)
        new{T, Nothing}(w0, C, nothing)
    end
end

function eval_exp_arg(m::Gaussian1D{<:Real, Nothing}, x)
    -(x/m.w0)^2
end

function eval_exp_arg(m::Gaussian1D{<:Real, <:NamedTuple}, x)
    d = m.data
    d.e_arg * x^2
end

function eval_mode(m::Gaussian1D{<:Real, Nothing}, x)
    m.C * exp(eval_exp_arg(m, x))
end

function eval_mode(m::Gaussian1D{<:Real, <:NamedTuple}, x)
    d = m.data
    m.C * exp(eval_exp_arg(m, x)) * d.eikz
end

"""
    Gaussian(w0::Real; norm_constant=nothing)
    Gaussian(w0x::Real, w0y::Real; norm_constant=nothing)
    Gaussian(w0::Real, λ::Real, z::Real; constant_phase=true, norm_constant=nothing)
    Gaussian(w0x::Real, w0y::Real, λ::Real, z::Real; constant_phase=true, norm_constant=nothing)

Create a two-dimensional Gaussian mode.

# Arguments
- `w0::Real`: Beam waist radius for circular beam
- `w0x::Real, w0y::Real`: Beam waist radii in x and y directions for elliptical beam
- `λ::Real`: Wavelength (for propagated version)
- `z::Real`: Propagation distance from waist (for propagated version)
- `constant_phase=true`: Include exp(ikz) phase factor
- `norm_constant=nothing`: Custom normalization constant - defines the peak intensity (default uses proper Gaussian normalization)

# Returns
`Gaussian` mode that can be evaluated at spatial coordinates.

# Examples
```jldoctest
julia> g = Gaussian(10.0);  # Circular beam, 50 μm waist

julia> nx, ny = 64, 64;

julia> dx, dy = 4.0, 4.0;

julia> xv, yv = spatial_vectors(nx, ny, dx, dy);

julia> field = zeros(ComplexF64, nx, ny);

julia> g(field, xv, yv);  # Evaluate on grid (in-place)

julia> sum(abs2, field) * dx * dy  # Check normalization
0.9999999999998386

julia> g_ellip = Gaussian(10.0, 20.0, 1.064, 1000.0);  # Elliptical beam, wavelength 1.064 µm, propagated 1 mm

julia> field_ellip = g_ellip(xv, yv);  # Direct evaluation (out-of-place)

julia> sum(abs2, field_ellip) * dx * dy
0.9999999999996259
```

See also: [`HermiteGaussian`](@ref), [`LaguerreGaussian`](@ref)
"""
struct Gaussian{T, G <: Gaussian1D{T}} <: Mode{2, T}
    gx::G
    gy::G

    function Gaussian(w0x::Real,
                      w0y::Real,
                      λ::Real,
                      z::Real;
                      constant_phase = true,
                      norm_constant = nothing)
        T = float(promote_type(typeof(w0x), typeof(w0y)))
        gx = Gaussian1D(T(w0x), λ, z; constant_phase, norm_constant)
        norm_constant = isnothing(norm_constant) ? nothing : 1.0
        # We don't want to account for the normalization constant twice
        gy = Gaussian1D(T(w0y), λ, z; constant_phase = false, norm_constant)
        # We don't want to account for constant phase twice
        new{T, typeof(gx)}(gx, gy)
    end

    function Gaussian(w0x::Real, w0y::Real; norm_constant = nothing)
        T = float(promote_type(typeof(w0x), typeof(w0y)))
        gx = Gaussian1D(w0x; norm_constant)
        norm_constant = isnothing(norm_constant) ? nothing : 1.0
        # We don't want to account for the normalization constant twice
        gy = Gaussian1D(w0y; norm_constant)
        new{T, typeof(gx)}(gx, gy)
    end

    function Gaussian(w0::Real,
                      λ::Real,
                      z::Real;
                      constant_phase = true,
                      norm_constant = nothing)
        Gaussian(w0, w0, λ, z; constant_phase, norm_constant)
    end

    function Gaussian(w0::Real; norm_constant = nothing)
        Gaussian(w0, w0; norm_constant)
    end
end

function eval_constant_phase(m::Gaussian1D{T, Nothing}) where {T <: Real}
    Complex{T}(1)
end

function eval_constant_phase(m::Gaussian1D{<:Real, <:NamedTuple})
    m.data.eikz
end

function eval_mode(m::Gaussian, x, y)
    Cx, Cy = m.gx.C, m.gy.C
    eikz = eval_constant_phase(m.gx)
    Cx * Cy * exp(eval_exp_arg(m.gx, x) + eval_exp_arg(m.gy, y)) * eikz
end

function hermite_polynomial(n::Integer)
    if n < 0
        throw(DomainError())
    elseif n == 0
        Polynomial([1])
    elseif n == 1
        Polynomial([0, 2])
    else
        2*Polynomial([0, 1])*hermite_polynomial(n-1)-2*(n-1)*hermite_polynomial(n-2)
    end
end

"""
    HermiteGaussian1D(w0::Real, n::Integer)
    HermiteGaussian1D(w0::Real, n::Integer, λ::Real, z::Real; constant_phase=true)

Create a one-dimensional Hermite-Gaussian mode HG_n.

# Arguments
- `w0::Real`: Beam waist radius
- `n::Integer`: Mode number (≥ 0)
- `λ::Real`: Wavelength (for propagated version)
- `z::Real`: Propagation distance from waist (for propagated version)
- `constant_phase=true`: Include exp(ikz) phase factor

# Returns
`HermiteGaussian1D` mode that can be evaluated at spatial positions.

# Examples
```jldoctest
julia> hg0 = HermiteGaussian1D(20.0, 0);  # HG_0 (Gaussian)

julia> hg1 = HermiteGaussian1D(20.0, 1);  # HG_1 (first excited mode)

julia> nx = 64;

julia> dx = 2.0;

julia> x, = spatial_vectors(nx, dx);

julia> field0 = hg0(x);

julia> field1 = hg1(x);

julia> sum(abs2, field0) * dx  # Check normalization
0.9999999998550136

julia> sum(field0 .* conj.(field1)) * dx  # Orthogonality check
-5.9951169107384464e-18 + 0.0im
```

See also: [`Gaussian1D`](@ref), [`Gaussian`](@ref), [`HermiteGaussian`](@ref), [`LaguerreGaussian`](@ref)
"""
struct HermiteGaussian1D{T, G <: Gaussian1D{T}, P} <: Mode{1, T}
    g::G
    C::Complex{T}
    hn::P
    n::Int

    function HermiteGaussian1D(w0::Real,
                               n::Integer,
                               λ::Real,
                               z::Real;
                               constant_phase = true)
        T = float(eltype(w0))
        g = Gaussian1D(w0, λ, z; constant_phase)
        hn = hermite_polynomial(n)
        qz = g.data.qz
        C = hg_normalization_constant(n)*(-conj(qz)/qz)^(n/2)
        new{T, typeof(g), typeof(hn)}(g, C, hn, n)
    end

    function HermiteGaussian1D(w0::Real, n::Integer)
        T = float(eltype(w0))
        g = Gaussian1D(w0)
        hn = hermite_polynomial(n)
        C = hg_normalization_constant(n)
        new{T, typeof(g), typeof(hn)}(g, C, hn, n)
    end
end

function eval_wz(m::HermiteGaussian1D{T, <:Gaussian1D{T, Nothing}}) where {T}
    m.g.w0
end

function eval_wz(m::HermiteGaussian1D{T, <:Gaussian1D{T, <:NamedTuple}}) where {T}
    m.g.data.wz
end

function eval_mode(m::HermiteGaussian1D, x)
    wz = eval_wz(m)
    m.C * m.hn(sqrt(2)*x/wz) * eval_mode(m.g, x)
end

"""
    HermiteGaussian(w0::Real, m::Integer, n::Integer)
    HermiteGaussian(w0x::Real, w0y::Real, m::Integer, n::Integer)
    HermiteGaussian(w0::Real, m::Integer, n::Integer, λ::Real, z::Real; constant_phase=true)
    HermiteGaussian(w0x::Real, w0y::Real, m::Integer, n::Integer, λ::Real, z::Real; constant_phase=true)

Create a two-dimensional Hermite-Gaussian mode HG_{m,n}.

# Arguments
- `w0::Real`: Beam waist radius for circular beam
- `w0x::Real, w0y::Real`: Beam waist radii in x and y directions
- `m::Integer, n::Integer`: Mode numbers in x and y directions (≥ 0)
- `λ::Real`: Wavelength (for propagated version)
- `z::Real`: Propagation distance from waist (for propagated version)
- `constant_phase=true`: Include exp(ikz) phase factor

# Returns
`HermiteGaussian` mode that can be evaluated at spatial coordinates.

# Examples
```jldoctest
julia> hg00 = HermiteGaussian(10.0, 0, 0);  # Fundamental mode

julia> hg10 = HermiteGaussian(10.0, 1, 0);  # First excited in x

julia> hg01 = HermiteGaussian(10.0, 0, 1);  # First excited in y

julia> nx, ny = 64, 64;

julia> dx, dy = 2.0, 2.0;

julia> xv, yv = spatial_vectors(nx, ny, dx, dy);

julia> field00 = hg00(xv, yv);

julia> field10 = hg10(xv, yv);

julia> sum(abs2, field00) * dx * dy  # Check normalization
0.9999999999999993

julia> sum(field00 .* conj.(field10)) * dx * dy  # Orthogonality
4.851917976626765e-18 + 0.0im
```

See also: [`Gaussian`](@ref), [`LaguerreGaussian`](@ref)
"""
struct HermiteGaussian{T, G <: HermiteGaussian1D{T}} <: Mode{2, T}
    hgx::G
    hgy::G

    function HermiteGaussian(w0x::Real,
                             w0y::Real,
                             m::Integer,
                             n::Integer,
                             λ::Real,
                             z::Real;
                             constant_phase = true)
        T = float(promote_type(typeof(w0x), typeof(w0y)))
        hgx = HermiteGaussian1D(T(w0x), m, λ, z; constant_phase)
        hgy = HermiteGaussian1D(T(w0y), n, λ, z; constant_phase = false)
        # We don't want to account for constant phase twice
        new{T, typeof(hgx)}(hgx, hgy)
    end

    function HermiteGaussian(w0x::Real, w0y::Real, m::Integer, n::Integer)
        T = float(promote_type(typeof(w0x), typeof(w0y)))
        hgx = HermiteGaussian1D(w0x, m)
        hgy = HermiteGaussian1D(w0y, n)
        new{T, typeof(hgx)}(hgx, hgy)
    end

    function HermiteGaussian(w0::Real,
                             m::Integer,
                             n::Integer,
                             λ::Real,
                             z::Real;
                             constant_phase = true)
        HermiteGaussian(w0, w0, m, n, λ, z; constant_phase)
    end

    function HermiteGaussian(w0::Real, m::Integer, n::Integer)
        HermiteGaussian(w0, w0, m, n)
    end
end

function eval_mode(m::HermiteGaussian, x, y)
    mx = m.hgx
    my = m.hgy
    wz_x = eval_wz(mx)
    wz_y = eval_wz(my)
    C = mx.C * mx.g.C * my.C * my.g.C
    eikz = eval_constant_phase(mx.g)
    (C *
     mx.hn(sqrt(2)*x/wz_x) *
     my.hn(sqrt(2)*y/wz_y) *
     exp(eval_exp_arg(mx.g, x) + eval_exp_arg(my.g, y)) *
     eikz)
end

"""
    hermite_gaussian_groups(w0, n_groups::Int)

Generate all Hermite-Gaussian modes up to a given group number.

Creates all HGₘₙ modes where m + n < n_groups, ordered by increasing total mode number.
This is useful for modal decomposition and beam shaping applications.

# Arguments
- `w0`: Beam waist radius
- `n_groups::Int`: Maximum group number

# Returns
Vector of `HermiteGaussian` modes.

# Examples
```jldoctest
julia> modes = hermite_gaussian_groups(10.0, 3);

julia> length(modes)  # Modes: HG_00, HG_10, HG_01, HG_20, HG_11, HG_02
6

julia> nx, ny = 64, 64;

julia> dx, dy = 2.0, 2.0;

julia> xv, yv = spatial_vectors(nx, ny, dx, dy);

julia> fields = [mode(xv, yv) for mode in modes];

julia> all(abs(sum(abs2, field) * dx * dy - 1.0) < 1e-10 for field in fields)  # All normalized
true

julia> modes = hermite_gaussian_groups(10.0, 4);

julia> length(modes)  # Groups 0, 1, 2, 3 give 1+2+3+4 = 10 modes
10
```

See also: [`Gaussian`](@ref), [`HermiteGaussian`](@ref), [`LaguerreGaussian`](@ref)
"""
function hermite_gaussian_groups(w0, n_groups::Int)
    @assert n_groups >= 0
    l = HermiteGaussian[]
    for m in 0:(n_groups - 1)
        for n in 0:(n_groups - m - 1)
            push!(l, HermiteGaussian(w0, m, n))
        end
    end
    l
end

function laguerre_polynomial(p::Integer, l::Integer)
    if p < 0
        throw(DomainError())
    elseif p == 0
        Polynomial([1])
    elseif p == 1
        Polynomial([1+l, -1])
    else
        ((2*p+l-1-Polynomial([0, 1]))*laguerre_polynomial(p-1, l) -
         (p+l-1)*laguerre_polynomial(p-2, l)) / p
    end
end

"""
    LaguerreGaussian(w0::Real, p::Integer, l::Integer; kind=:vortex)
    LaguerreGaussian(w0::Real, p::Integer, l::Integer, λ::Real, z::Real; constant_phase=true, kind=:vortex)

Create a Laguerre-Gaussian mode LGₚₗ.

# Arguments
- `w0::Real`: Beam waist radius
- `p::Integer`: Radial mode number (≥ 0)
- `l::Integer`: Azimuthal mode number (any integer)
- `λ::Real`: Wavelength (for propagated version)
- `z::Real`: Propagation distance from waist (for propagated version)
- `constant_phase=true`: Include exp(ikz) phase factor
- `kind`: Mode type - `:vortex` (default), `:even`, or `:odd`
  - `:vortex`: exp(ilφ) phase dependence
  - `:even`: cos(lφ) dependence  
  - `:odd`: sin(lφ) dependence

# Returns
`LaguerreGaussian` mode that can be evaluated at spatial coordinates.

# Examples
```jldoctest
julia> lg00 = LaguerreGaussian(10.0, 0, 0);  # Fundamental mode (Gaussian)

julia> lg01 = LaguerreGaussian(10.0, 0, 1);  # Vortex beam

julia> nx, ny = 65, 65;

julia> dx, dy = 2.0, 2.0;

julia> xv, yv = spatial_vectors(nx, ny, dx, dy);

julia> field00 = lg00(xv, yv);

julia> field01 = lg01(xv, yv);

julia> sum(abs2, field00) * dx * dy  # Check normalization
0.9999999999999998

julia> abs(field01[nx÷2+1, ny÷2+1]) < 1e-10  # LG_01 has zero at center (vortex)
true

julia> lg_even = LaguerreGaussian(10.0, 0, 2; kind=:even);

julia> field_even = lg_even(xv, yv);

julia> all(isreal, field_even)  # Even modes are real-valued
true
```

See also: [`Gaussian`](@ref), [`HermiteGaussian`](@ref)
"""
struct LaguerreGaussian{T, P <: Union{Nothing, <:NamedTuple}, K, L} <: Mode{2, T}
    w0::T
    C::Complex{T}
    lp::L
    order::@NamedTuple{p::Int, l::Int}
    data::P
    kind::K

    function LaguerreGaussian(w0::Real,
                              p::Integer,
                              l::Integer,
                              λ::Real,
                              z::Real;
                              constant_phase = true,
                              kind = :vortex)
        @assert p >= 0
        T = float(eltype(w0))
        q0 = Complex{T}(get_q(w0, λ, 0))
        qz = Complex{T}(get_q(w0, λ, z))
        wz = Complex{T}(get_w(w0, λ, z))
        k = T(2π/λ)
        eikz = constant_phase ? Complex{T}(exp(im*k*z)) : Complex{T}(1)
        C = lg_normalization_constant(w0, p, l) * (q0/qz) * (-conj(qz)/qz)^(p+abs(l)/2)
        lp = laguerre_polynomial(p, abs(l))
        data = (λ = λ, z = z, wz = wz, eikz = eikz, e_arg = im*k/(2*qz))
        kind = parse_kind(kind)
        P = typeof(data)
        L = typeof(lp)
        K = typeof(kind)
        new{T, P, K, L}(w0, C, lp, (p = p, l = l), data, kind)
    end

    function LaguerreGaussian(w0::Real, p::Integer, l::Integer; kind = :vortex)
        T = float(eltype(w0))
        C = Complex{T}(lg_normalization_constant(w0, p, l))
        lp = laguerre_polynomial(p, abs(l))
        kind = parse_kind(kind)
        L = typeof(lp)
        K = typeof(kind)
        new{T, Nothing, K, L}(w0, C, lp, (p = p, l = l), nothing, kind)
    end
end

function Base.eltype(m::LaguerreGaussian{T}) where {T}
    Complex{T}
end

function eval_lg_arg(m::LaguerreGaussian{T, P, Vortex}, x, y) where {T, P}
    l = m.order.l
    exp(im*l*atan(y, x))
end

function eval_lg_arg(m::LaguerreGaussian{T, P, Even}, x, y) where {T, P}
    l = m.order.l
    sqrt(2)*cos(l*atan(y, x))
end

function eval_lg_arg(m::LaguerreGaussian{T, P, Odd}, x, y) where {T, P}
    l = m.order.l
    sqrt(2)*sin(l*atan(y, x))
end

function eval_mode(m::LaguerreGaussian{T, Nothing}, x, y) where {T}
    l = m.order.l
    r2 = (x^2 + y^2) / m.w0^2
    rl = (sqrt(2)*sqrt(r2))^abs(l)
    m.C * rl * m.lp(2*r2) * exp(-r2) * eval_lg_arg(m, x, y)
end

function eval_mode(m::LaguerreGaussian{T, <:NamedTuple}, x, y) where {T}
    l = m.order.l
    wz = m.data.wz
    r2 = x^2 + y^2
    rl = (sqrt(2)*sqrt(r2)/wz)^abs(l)
    m.C * rl * m.lp(2*r2/wz^2) * exp(m.data.e_arg*r2) * eval_lg_arg(m, x, y) * m.data.eikz
end
