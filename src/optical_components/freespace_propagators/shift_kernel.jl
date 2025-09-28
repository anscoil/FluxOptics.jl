function shift_kernel(fx::T, fy::T, θx::T, θy::T, z::Tp) where {T <: Real, Tp <: Real}
    fx, fy = Tp(fx), Tp(fy)
    θx, θy = Tp(θx), Tp(θy)
    f0x, f0y = tan(θx), tan(θy)
    Complex{T}(cis(-2π*z*(f0x*fx + f0y*fy)))
end

function shift_kernel(fx::T, θx::T, z::Tp) where {T <: Real, Tp <: Real}
    fx, θx = Tp(fx), Tp(θx)
    f0x = tan(θx)
    Complex{T}(cis(-2π*z*f0x*fx))
end

struct ShiftKernel{M, K, T, Tp} <: AbstractPropagator{M, K, T}
    kernel::K
    z::Tp

    function ShiftKernel(u::ScalarField{U, Nd},
                         ds::NTuple{Nd, Real},
                         z::Real;
                         use_cache::Bool = true,
                         double_precision_kernel::Bool = use_cache) where {Nd, T,
                                                                           U <:
                                                                           AbstractArray{Complex{T}}}
        ns = size(u)[1:Nd]
        cache_size = use_cache ? prod(size(u)[(Nd + 1):end]) : 0
        kernel = FourierKernel(u.electric, ns, ds, cache_size)
        Tp = double_precision_kernel ? Float64 : T
        new{Static, typeof(kernel), T, Tp}(kernel, Tp(z))
    end
end

Functors.@functor ShiftKernel ()

get_kernels(p::ShiftKernel) = (p.kernel,)

build_kernel_key_args(p::ShiftKernel, u::ScalarField) = (select_tilts(u)...,)

build_kernel_args(p::ShiftKernel) = (p.z,)

function _propagate_core!(apply_kernel_fns::F,
                          u::AbstractArray,
                          p::ShiftKernel,
                          ::Type{<:Direction}) where {F}
    apply_kernel_fn!, = apply_kernel_fns
    apply_kernel_fn!(u, shift_kernel)
    u
end

struct ShiftProp{M, C} <: AbstractSequence{M}
    optical_components::C

    function ShiftProp(u::ScalarField{U, Nd},
                       ds::NTuple{Nd, Real},
                       z::Real;
                       use_cache::Bool = true,
                       double_precision_kernel::Bool = use_cache) where {U, Nd}
        kernel = ShiftKernel(u, ds, z; use_cache, double_precision_kernel)
        wrapper = FourierWrapper(kernel.kernel.p_f, kernel)
        M = get_trainability(wrapper)
        optical_components = get_sequence(wrapper)
        C = typeof(optical_components)
        new{M, C}(optical_components)
    end

    function ShiftProp(u::ScalarField,
                       z::Real;
                       use_cache::Bool = true,
                       double_precision_kernel::Bool = use_cache)
        ShiftProp(u, u.ds, z; use_cache, double_precision_kernel)
    end
end

Functors.@functor ShiftProp (optical_components,)

get_sequence(p::ShiftProp) = p.optical_components
