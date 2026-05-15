"""
Levenberg-Marquardt BRDF optimizer.

Fits analytical BRDF models to measured data by minimising the sum-of-squared
residuals between model predictions and measured values over a fixed set of
sampling directions.

Uses LsqFit.jl's `curve_fit` (a box-constrained LM implementation) to jointly
estimate all free parameters of a given `BRDFModel` type.

# Usage
```julia
data  = load_brdf("path/to/material.binary")
model = fit_brdf(LevenbergMarquardt(), CookTorranceModel, data)
```

Per-model `fit_brdf` methods handle parameter packing, bounds, and initial
guesses. All methods share the same sampling strategy and residual computation
via the internal `_fit_lm` helper.
"""

using LsqFit
using StaticArrays
using Colors
using LinearAlgebra

# ============================================================================
# Optimizer Type
# ============================================================================

"""
    LevenbergMarquardt <: BRDFOptimizer

Levenberg-Marquardt fitter (via LsqFit.jl).

# Fields
- `n_samples`:     Number of (wi, wo) direction pairs to fit against (default 1024).
- `max_iter`:      Maximum LM iterations (default 200).
- `x_tol`:         Parameter convergence tolerance (default 1e-6).
- `g_tol`:         Gradient convergence tolerance (default 1e-8).
- `cosine_weight`: If `true`, multiply residuals by `cos(θi)` to down-weight
                   grazing-angle noise (default `true`).
- `loss`:          Residual transform. `:log` fits `log1p(f)` (default — handles
                   large specular/diffuse dynamic range). `:linear` fits raw `f`.
- `sampling`:      Direction sampling strategy. `:rusinkiewicz` (default) samples
                   in `(θ_h, φ_h, θ_d, φ_d)` space so the specular lobe is
                   densely covered. `:uniform` samples `(wi, wo)` independently
                   on the hemisphere (poor for sharp specular lobes).
"""
struct LevenbergMarquardt <: BRDFOptimizer
    n_samples::Int
    max_iter::Int
    x_tol::Float64
    g_tol::Float64
    cosine_weight::Bool
    loss::Symbol
    sampling::Symbol
end

function LevenbergMarquardt(;
    n_samples::Integer = 1024,
    max_iter::Integer = 200,
    x_tol::Real = 1e-6,
    g_tol::Real = 1e-8,
    cosine_weight::Bool = true,
    loss::Symbol = :log,
    sampling::Symbol = :rusinkiewicz,
)
    loss in (:log, :linear) ||
        throw(ArgumentError("loss must be :log or :linear, got $loss"))
    sampling in (:rusinkiewicz, :uniform) ||
        throw(ArgumentError("sampling must be :rusinkiewicz or :uniform, got $sampling"))
    LevenbergMarquardt(Int(n_samples), Int(max_iter), Float64(x_tol),
                       Float64(g_tol), cosine_weight, loss, sampling)
end

# ============================================================================
# Sampling Directions
# ============================================================================

"""
    _halton(i, base) -> Float64

Radical-inverse Halton sequence sample (low-discrepancy 1D quasi-random).
"""
@inline function _halton(i::Int, base::Int)::Float64
    f = 1.0
    r = 0.0
    j = i
    while j > 0
        f /= base
        r += f * (j % base)
        j ÷= base
    end
    return r
end

"""
    _sample_directions_uniform(n) -> (wis, wos)

Independent cosine-weighted hemisphere sampling of wi and wo (golden-ratio
quasi-random). Poor for sharp specular lobes — use only as a baseline.
"""
function _sample_directions_uniform(n::Int)
    wis = Vector{SVector{3, Float32}}(undef, n)
    wos = Vector{SVector{3, Float32}}(undef, n)

    g1 = 0.7548776662466927    # 1 / φ
    g2 = 0.5698402909980532    # 1 / φ²
    for k in 1:n
        u1 = mod(k * g1, 1.0); u2 = mod(k * g2, 1.0)
        r = sqrt(u1); phi_i = 2π * u2
        wis[k] = SVector{3, Float32}(r * cos(phi_i), r * sin(phi_i),
                                     sqrt(max(0.0, 1.0 - u1)))

        v1 = mod((k + 0.5) * g1 + 0.3, 1.0); v2 = mod((k + 0.5) * g2 + 0.7, 1.0)
        r2 = sqrt(v1); phi_o = 2π * v2
        wos[k] = SVector{3, Float32}(r2 * cos(phi_o), r2 * sin(phi_o),
                                     sqrt(max(0.0, 1.0 - v1)))
    end
    return wis, wos
end

"""
    _sample_directions_rusinkiewicz(n) -> (wis, wos)

Sample `(wi, wo)` pairs in Rusinkiewicz half-angle space `(θ_h, φ_h, θ_d, φ_d)`.

The key property: `θ_h = 0` corresponds to the perfect mirror direction, so
biasing samples towards small `θ_h` (via a quadratic remap of the Halton
sequence) puts dense coverage on the specular lobe — exactly where a sharp
microfacet BRDF needs it. This matches MERL's native storage layout.

Invalid samples (where wi or wo drops below the horizon) are replaced with
the next valid Halton draw.
"""
function _sample_directions_rusinkiewicz(n::Int)
    wis = Vector{SVector{3, Float32}}(undef, n)
    wos = Vector{SVector{3, Float32}}(undef, n)

    N = SVector{3, Float32}(0.0f0, 0.0f0, 1.0f0)
    k = 1
    idx = 0
    max_tries = 50 * n
    while k <= n && idx < max_tries
        idx += 1
        u1 = _halton(idx, 2)
        u2 = _halton(idx, 3)
        u3 = _halton(idx, 5)
        u4 = _halton(idx, 7)

        # Quadratic remap biases θ_h near 0 (mirrors MERL's sqrt index mapping)
        theta_h = Float32(u1 * u1 * (π / 2))
        phi_h   = Float32(2π * u2)
        theta_d = Float32(u3 * (π / 2))
        phi_d   = Float32(π * u4)      # [0, π]: exploit isotropic reciprocity

        sth, cth = sin(theta_h), cos(theta_h)
        sph, cph = sin(phi_h),   cos(phi_h)
        h = SVector{3, Float32}(sth * cph, sth * sph, cth)

        # Build frame (t, b, h) matching the MERL loader convention
        Nh = dot(N, h)
        t_raw = N - Nh * h
        tlen = norm(t_raw)
        local t, b
        if tlen > 1.0f-6
            t = t_raw / tlen
            b = cross(h, t)
        else
            t = SVector{3, Float32}(1.0f0, 0.0f0, 0.0f0)
            b = SVector{3, Float32}(0.0f0, 1.0f0, 0.0f0)
        end

        std, ctd = sin(theta_d), cos(theta_d)
        spd, cpd = sin(phi_d),   cos(phi_d)
        wl = SVector{3, Float32}(std * cpd, std * spd, ctd)   # wi in h-frame

        wi = SVector{3, Float32}(
            wl[1]*t[1] + wl[2]*b[1] + wl[3]*h[1],
            wl[1]*t[2] + wl[2]*b[2] + wl[3]*h[2],
            wl[1]*t[3] + wl[2]*b[3] + wl[3]*h[3],
        )
        wh = dot(wi, h)
        wo = SVector{3, Float32}(
            2*wh*h[1] - wi[1],
            2*wh*h[2] - wi[2],
            2*wh*h[3] - wi[3],
        )

        if wi[3] > 0.0f0 && wo[3] > 0.0f0
            wis[k] = wi
            wos[k] = wo
            k += 1
        end
    end
    # If we ran out of Halton draws (very unlikely), pad with normal-incidence
    while k <= n
        wis[k] = SVector{3, Float32}(0, 0, 1)
        wos[k] = SVector{3, Float32}(0, 0, 1)
        k += 1
    end
    return wis, wos
end

"""
    _sample_directions(opt) -> (wis, wos)

Dispatch sampling strategy based on `opt.sampling`.
"""
function _sample_directions(opt::LevenbergMarquardt)
    if opt.sampling === :rusinkiewicz
        return _sample_directions_rusinkiewicz(opt.n_samples)
    else
        return _sample_directions_uniform(opt.n_samples)
    end
end

# ============================================================================
# Shared Fit Driver
# ============================================================================

"""
    _fit_lm(opt, reconstruct, p0, lower, upper, data) -> (params, model, result)

Generic LM driver used by all per-model `fit_brdf` methods.

# Arguments
- `opt`:         Optimizer settings.
- `reconstruct`: Function `params -> BRDFModel`, building a concrete model
                 from a parameter vector.
- `p0`:          Initial parameter vector (`Vector{Float64}`).
- `lower`:       Lower bounds (same length as p0).
- `upper`:       Upper bounds (same length as p0).
- `data`:        Target `MeasuredBRDF`.

Returns `(fitted_params, fitted_model, lsq_result)`.
"""
function _fit_lm(
    opt::LevenbergMarquardt,
    reconstruct::Function,
    p0::Vector{Float64},
    lower::Vector{Float64},
    upper::Vector{Float64},
    data::MeasuredBRDF,
)
    # --- Build target vector from measured data ---
    wis, wos = _sample_directions(opt)

    # Residual transform: log1p compresses the large specular/diffuse dynamic
    # range so both regimes contribute meaningfully to the LM gradient.
    use_log = opt.loss === :log
    transform(x) = use_log ? log1p(max(x, 0.0)) : x

    targets = Vector{Float64}(undef, 3 * opt.n_samples)
    for k in 1:opt.n_samples
        c = lookup(data, wis[k], wos[k])
        w = opt.cosine_weight ? Float64(wis[k][3]) : 1.0
        targets[3k - 2] = transform(Float64(c.r) * w)
        targets[3k - 1] = transform(Float64(c.g) * w)
        targets[3k    ] = transform(Float64(c.b) * w)
    end

    # --- Model function: params -> predicted values ---
    function model_fn(_xs, p)
        m = reconstruct(p)
        preds = Vector{Float64}(undef, 3 * opt.n_samples)
        for k in 1:opt.n_samples
            c = evaluate(m, wis[k], wos[k])
            w = opt.cosine_weight ? Float64(wis[k][3]) : 1.0
            preds[3k - 2] = transform(Float64(c.r) * w)
            preds[3k - 1] = transform(Float64(c.g) * w)
            preds[3k    ] = transform(Float64(c.b) * w)
        end
        return preds
    end

    xdummy = collect(1:length(targets))

    result = curve_fit(
        model_fn, xdummy, targets, p0;
        lower = lower,
        upper = upper,
        maxIter = opt.max_iter,
        x_tol  = opt.x_tol,
        g_tol  = opt.g_tol,
    )

    fitted_params = result.param
    fitted_model  = reconstruct(fitted_params)
    return fitted_params, fitted_model, result
end

# ============================================================================
# Per-Model fit_brdf Methods
# ============================================================================

# Parameter bounds shared across microfacet models
const _ROUGH_LO = 0.02
const _ROUGH_HI = 1.0
const _F0_LO    = 0.0
const _F0_HI    = 1.0
const _ALB_LO   = 0.0
const _ALB_HI   = 1.0
const _KD_LO    = 0.0
const _KD_HI    = 1.0

"""
    fit_brdf(opt::LevenbergMarquardt, ::Type{LambertianModel}, data::MeasuredBRDF)

Fit a `LambertianModel` (3 parameters: RGB albedo).
"""
function fit_brdf(opt::LevenbergMarquardt,
                  ::Type{LambertianModel},
                  data::MeasuredBRDF)::LambertianModel
    reconstruct(p) = LambertianModel(RGB{Float32}(p[1], p[2], p[3]))
    p0    = [0.5, 0.5, 0.5]
    lower = [_ALB_LO, _ALB_LO, _ALB_LO]
    upper = [_ALB_HI, _ALB_HI, _ALB_HI]
    _, model, _ = _fit_lm(opt, reconstruct, p0, lower, upper, data)
    return model
end

# -- 7-parameter microfacet + diffuse: (roughness, F0rgb, albedo_rgb) ---------

function _fit_microfacet_diffuse(opt, ModelType, data)
    reconstruct(p) = ModelType(
        Float32(p[1]),
        RGB{Float32}(p[2], p[3], p[4]),
        RGB{Float32}(p[5], p[6], p[7]),
    )
    p0    = [0.1,       0.04, 0.04, 0.04,   0.3, 0.3, 0.3]
    lower = [_ROUGH_LO, _F0_LO, _F0_LO, _F0_LO,   _ALB_LO, _ALB_LO, _ALB_LO]
    upper = [_ROUGH_HI, _F0_HI, _F0_HI, _F0_HI,   _ALB_HI, _ALB_HI, _ALB_HI]
    _, model, _ = _fit_lm(opt, reconstruct, p0, lower, upper, data)
    return model
end

fit_brdf(opt::LevenbergMarquardt, ::Type{CookTorranceModel}, data::MeasuredBRDF) =
    _fit_microfacet_diffuse(opt, CookTorranceModel, data)

fit_brdf(opt::LevenbergMarquardt, ::Type{CookTorranceFresnelDiffuseModel}, data::MeasuredBRDF) =
    _fit_microfacet_diffuse(opt, CookTorranceFresnelDiffuseModel, data)

fit_brdf(opt::LevenbergMarquardt, ::Type{GGXModel}, data::MeasuredBRDF) =
    _fit_microfacet_diffuse(opt, GGXModel, data)

fit_brdf(opt::LevenbergMarquardt, ::Type{GGXFresnelDiffuseModel}, data::MeasuredBRDF) =
    _fit_microfacet_diffuse(opt, GGXFresnelDiffuseModel, data)

# -- 4-parameter pure-specular: (roughness, F0rgb) ---------------------------

function _fit_microfacet_specular(opt, ModelType, data)
    reconstruct(p) = ModelType(
        Float32(p[1]),
        RGB{Float32}(p[2], p[3], p[4]),
    )
    p0    = [0.1,       0.04, 0.04, 0.04]
    lower = [_ROUGH_LO, _F0_LO, _F0_LO, _F0_LO]
    upper = [_ROUGH_HI, _F0_HI, _F0_HI, _F0_HI]
    _, model, _ = _fit_lm(opt, reconstruct, p0, lower, upper, data)
    return model
end

fit_brdf(opt::LevenbergMarquardt, ::Type{CookTorranceSpecularModel}, data::MeasuredBRDF) =
    _fit_microfacet_specular(opt, CookTorranceSpecularModel, data)

fit_brdf(opt::LevenbergMarquardt, ::Type{GGXSpecularModel}, data::MeasuredBRDF) =
    _fit_microfacet_specular(opt, GGXSpecularModel, data)

# -- 8-parameter scaled-diffuse: (roughness, F0rgb, albedo_rgb, kd) ----------

function _fit_microfacet_scaled(opt, ModelType, data)
    reconstruct(p) = ModelType(
        Float32(p[1]),
        RGB{Float32}(p[2], p[3], p[4]),
        RGB{Float32}(p[5], p[6], p[7]),
        Float32(p[8]),
    )
    p0    = [0.1,       0.04, 0.04, 0.04,   0.3, 0.3, 0.3,   0.5]
    lower = [_ROUGH_LO, _F0_LO, _F0_LO, _F0_LO,   _ALB_LO, _ALB_LO, _ALB_LO,   _KD_LO]
    upper = [_ROUGH_HI, _F0_HI, _F0_HI, _F0_HI,   _ALB_HI, _ALB_HI, _ALB_HI,   _KD_HI]
    _, model, _ = _fit_lm(opt, reconstruct, p0, lower, upper, data)
    return model
end

fit_brdf(opt::LevenbergMarquardt, ::Type{CookTorranceScaledDiffuseModel}, data::MeasuredBRDF) =
    _fit_microfacet_scaled(opt, CookTorranceScaledDiffuseModel, data)

fit_brdf(opt::LevenbergMarquardt, ::Type{GGXScaledDiffuseModel}, data::MeasuredBRDF) =
    _fit_microfacet_scaled(opt, GGXScaledDiffuseModel, data)

# ============================================================================
# Option declaration (introspected by the UI and future HP tuners)
# ============================================================================

optimizer_options(::Type{LevenbergMarquardt}) = OptimizerOption[
    OptimizerOption(:sampling, "Sampling", :choice, :rusinkiewicz,
                    Any[:rusinkiewicz, :uniform], (0, 0), nothing,
                    "Direction sampling strategy"),
    OptimizerOption(:n_samples, "Samples", :int, 1024,
                    Any[], (256, 8192), 256,
                    "Number of (wi, wo) sample pairs"),
    OptimizerOption(:cosine_weight, "Cosine weight", :bool, true,
                    Any[], (0, 0), nothing,
                    "Down-weight grazing-angle residuals"),
    OptimizerOption(:loss, "Loss", :choice, :log,
                    Any[:log, :linear], (0, 0), nothing,
                    "Residual transform (:log handles HDR specular)"),
]

# ============================================================================
# Registration
# ============================================================================

export LevenbergMarquardt, fit_brdf
register_optimizer!(LevenbergMarquardt)
