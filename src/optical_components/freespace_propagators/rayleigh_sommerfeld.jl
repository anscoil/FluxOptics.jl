function rs_kernel(x::T, y::T, λ::T, z::Tp, nrm_f::Tp,
                   z_pos::Val{true}) where {T <: Real, Tp <: Real}
    x, y = Tp(x), Tp(y)
    k = Tp(2π/λ)
    r = sqrt(x^2 + y^2 + z^2)
    Complex{T}(nrm_f*(cis(k*r)/r)*(z/r)*(1/r-im*k))
end

function rs_kernel(x::T, y::T, λ::T, z::Tp, nrm_f::Tp,
                   z_pos::Val{false}) where {T <: Real, Tp <: Real}
    conj(rs_kernel(x, y, λ, -z, nrm_f, Val(true)))
end

function rs_tilted_kernel(x::T, y::T, λ::T, θx::T, θy::T, track_tilts::Bool, z::Tp,
                          nrm_f::Tp, z_pos::Val{true}) where {T <: Real, Tp <: Real}
    x, y = Tp(x), Tp(y)
    f0x, f0y = sin(θx)/λ, sin(θy)/λ
    k = Tp(2π/λ)
    r = sqrt(x^2 + y^2 + z^2)
    lin_phase = track_tilts ? Complex{Tp}(1) : cis(-Tp(2)*π*(x*f0x+y*f0y))
    Complex{T}(nrm_f*(cis(k*r)/r)*lin_phase*(z/r)*(1/r-im*k))
end

function rs_tilted_kernel(x::T, y::T, λ::T, θx::T, θy::T, track_tilts::Bool,
                          z::Tp, nrm_f::Tp, z_pos::Val{false}) where {T <: Real, Tp <: Real}
    conj(rs_tilted_kernel(x, y, λ, θx, θy, track_tilts, -z, nrm_f, Val(true)))
end

function rs_valid_distance(nx, ny, dx, dy, λ)
    if dx < λ/2 && dy < λ/2
        return 0.0
    end
    zc_x = dx < λ/2 ? 0.0 : (nx*dx/2) * sqrt(4*dx^2/λ^2 - 1)
    zc_y = dy < λ/2 ? 0.0 : (ny*dy/2) * sqrt(4*dy^2/λ^2 - 1)
    max(zc_x, zc_y)
end

struct RSKernelProp{M, K, T, Tp} <: AbstractPropagator{M, K, T}
    kernel::K
    track_tilts::Bool
    z::Tp
    nrm_f::Tp

    function RSKernelProp(u::ScalarField{U, Nd},
                          ds::NTuple{Nd, Real},
                          z::Real;
                          use_cache::Bool = true,
                          track_tilts::Bool = false,
                          double_precision_kernel::Bool
                          = use_cache) where {T, U <: AbstractArray{Complex{T}}, Nd}
        ns = size(u)[1:Nd]
        cache_size = use_cache ? prod(size(u)[(Nd + 1):end]) : 0
        kernel = ConvolutionKernel(u.electric, ns, ds, cache_size)
        Tp = double_precision_kernel ? Float64 : T
        nrm_f = Tp(prod(ds)/2π)
        new{Static, typeof(kernel), T, Tp}(kernel, track_tilts, Tp(z), nrm_f)
    end
end

Functors.@functor RSKernelProp ()

get_kernels(p::RSKernelProp) = (p.kernel,)

function build_kernel_key_args(p::RSKernelProp, u::ScalarField)
    if is_on_axis(u)
        (select_lambdas(u),)
    else
        (select_lambdas(u), select_tilts(u)...)
    end
end

function build_kernel_args(p::RSKernelProp, u::ScalarField)
    if is_on_axis(u)
        (p.z, p.nrm_f, Val(sign(p.z) > 0))
    else
        (p.track_tilts, p.z, p.nrm_f, Val(sign(p.z) > 0))
    end
end

function _propagate_core!(apply_kernel_fns::F,
                          u::ScalarField,
                          p::RSKernelProp,
                          ::Type{<:Direction}) where {F}
    apply_kernel_fn!, = apply_kernel_fns
    if is_on_axis(u)
        apply_kernel_fn!(u.electric, rs_kernel)
    else
        apply_kernel_fn!(u.electric, rs_tilted_kernel)
    end
    u
end

struct RSProp{M, C} <: AbstractSequence{M}
    optical_components::C

    function RSProp(optical_components::C) where {C}
        new{Trainable, C}(optical_components)
    end

    function RSProp(u::ScalarField{U, Nd},
                    ds::NTuple{Nd, Real},
                    z::Real;
                    use_cache::Bool = true,
                    track_tilts::Bool = false,
                    double_precision_kernel::Bool
                    = use_cache) where {T, U <: AbstractArray{Complex{T}}, Nd}
        ns = size(u)[1:Nd]
        zc = rs_valid_distance(ns..., ds..., minimum(u.lambdas.collection))
        if abs(z) < zc
            @warn """RSProp: propagation distance z=$z is below critical distance zc=$zc.
             Numerical artifacts expected. Consider using ASProp or finer sampling (dx < λ/2)."""
        end
        rs = RSKernelProp(u, ds, z; use_cache, track_tilts, double_precision_kernel)
        wrapper = FourierWrapper(rs.kernel.p_f, rs)
        pad_op = PadCropOperator(u, rs.kernel.u_plan; store_ref = true)
        crop_op = adjoint(pad_op)
        optical_components = (pad_op, get_sequence(wrapper)..., crop_op)
        M = get_trainability(wrapper)
        C = typeof(optical_components)
        new{M, C}(optical_components)
    end

    function RSProp(u::ScalarField,
                    z::Real;
                    use_cache::Bool = true,
                    track_tilts::Bool = false,
                    double_precision_kernel::Bool = use_cache)
        RSProp(u, Tuple(u.ds), z; use_cache, track_tilts, double_precision_kernel)
    end
end

Functors.@functor RSProp (optical_components,)

get_sequence(p::RSProp) = p.optical_components
