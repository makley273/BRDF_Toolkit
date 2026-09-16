"""
Levenberg-Marquardt BRDF optimizer.

Fits analytical BRDF models to measured data by minimising the sum-of-squared
residuals between model predictions and measured values over a fixed set of
sampling directions.

Uses LsqFit.jl's `curve_fit` (a box-constrained LM implementation) to jointly
estimate all free parameters of a given `BRDFModel` type.

Sampling, the residual transform, per-model parameter packing, and multi-start
initialisation are shared with the other optimizers via `_fit_common.jl`.

# Usage
```julia
data  = load_brdf("path/to/material.binary")
model = fit_brdf(LevenbergMarquardt(), CookTorranceModel, data)
```
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
- `loss`:          Loss function (`:log`, `:linear`, `:cuberoot`, `:ngan`,
                   `:huber`, `:logcosh`). `:log` (default) handles the large
                   specular/diffuse dynamic range and matches the headline
                   RMS-LogWE metric; `:cuberoot` matches RMS-CRWE; `:ngan` is a
                   relative error metric.
- `sampling`:      Direction sampling strategy (`:rusinkiewicz`, `:uniform_half`,
                   `:mixed`, `:uniform`). See `_sample_directions`.
- `n_starts`:      Number of initial guesses to try (default 1). Values > 1 run
                   a deterministic multi-start and keep the best fit, guarding
                   against poor local minima.
"""
struct LevenbergMarquardt <: BRDFOptimizer
    n_samples::Int
    max_iter::Int
    x_tol::Float64
    g_tol::Float64
    cosine_weight::Bool
    loss::Symbol
    sampling::Symbol
    n_starts::Int
end

function LevenbergMarquardt(;
    n_samples::Integer = 1024,
    max_iter::Integer = 200,
    x_tol::Real = 1e-6,
    g_tol::Real = 1e-8,
    cosine_weight::Bool = true,
    loss::Symbol = :log,
    sampling::Symbol = :uniform_cosine,
    n_starts::Integer = 1,
)
    loss in (:log, :linear, :cuberoot, :ngan, :huber, :logcosh) ||
        throw(ArgumentError("loss must be :log, :linear, :cuberoot, :ngan, :huber, or :logcosh"))
    sampling in VALID_SAMPLINGS ||
        throw(ArgumentError("sampling must be one of $(VALID_SAMPLINGS); got $sampling"))
    n_starts >= 1 || throw(ArgumentError("n_starts must be >= 1, got $n_starts"))
    LevenbergMarquardt(Int(n_samples), Int(max_iter), Float64(x_tol),
                       Float64(g_tol), cosine_weight, loss, sampling, Int(n_starts))
end

# ============================================================================
# Fit Driver
# ============================================================================

"""
    fit_brdf(opt::LevenbergMarquardt, ModelType, data) -> BRDFModel

Fit any model with a registered `_fit_spec` (see `_fit_common.jl`). Runs
`opt.n_starts` Levenberg-Marquardt fits from spread initial guesses and returns
the model with the lowest sum-of-squared residuals.
"""
function fit_brdf(opt::LevenbergMarquardt,
                  ::Type{M},
                  data::MeasuredBRDF)::M where {M <: BRDFModel}
    spec        = _fit_spec(M)
    wis, wos    = _sample_directions(opt.sampling, opt.n_samples)
    residual_fn = _residual_function(opt.loss)
    targets     = _fit_targets(data, wis, wos, opt.cosine_weight)
    xdummy      = collect(1:length(targets))

    # curve_fit computes (model_fn - ydata); we make model_fn return the
    # per-sample pseudo-residual directly and fit it against a zero vector, so
    # the squared LM objective equals sum(residual_fn(pred, target)^2).
    zeros_y = zeros(length(targets))
    function model_fn(_xs, p)
        preds = _fit_predict(spec.reconstruct(p), wis, wos, opt.cosine_weight)
        residual_fn.(preds, targets)
    end

    starts   = _generate_starts(spec.p0, spec.lower, spec.upper, opt.n_starts)
    best_param = spec.p0
    best_obj   = Inf
    for p0 in starts
        result = curve_fit(
            model_fn, xdummy, zeros_y, p0;
            lower = spec.lower,
            upper = spec.upper,
            maxIter = opt.max_iter,
            x_tol  = opt.x_tol,
            g_tol  = opt.g_tol,
        )
        obj = sum(abs2, result.resid)
        if obj < best_obj
            best_obj   = obj
            best_param = result.param
        end
    end
    return spec.reconstruct(best_param)
end

# ============================================================================
# Option declaration (introspected by the UI and future HP tuners)
# ============================================================================

optimizer_options(::Type{LevenbergMarquardt}) = OptimizerOption[
    OptimizerOption(:sampling, "Sampling", :choice, :uniform_cosine,
                    Any[VALID_SAMPLINGS...], (0, 0), nothing,
                    "Direction sampling strategy"),
    OptimizerOption(:n_samples, "Samples", :int, 1024,
                    Any[], (256, 8192), 256,
                    "Number of (wi, wo) sample pairs"),
    OptimizerOption(:cosine_weight, "Cosine weight", :bool, true,
                    Any[], (0, 0), nothing,
                    "Down-weight grazing-angle residuals"),
    OptimizerOption(:loss, "Loss", :choice, :log,
                    Any[:log, :linear, :cuberoot, :ngan, :huber, :logcosh], (0, 0), nothing,
                    "Loss function (:log HDR, :ngan relative, :huber robust)"),
    OptimizerOption(:n_starts, "Restarts", :int, 1,
                    Any[], (1, 16), 1,
                    "Multi-start initial guesses; keeps the best fit"),
]

# ============================================================================
# Registration
# ============================================================================

export LevenbergMarquardt, fit_brdf
register_optimizer!(LevenbergMarquardt)
