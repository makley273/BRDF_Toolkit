"""
Nelder-Mead BRDF optimizer.

Gradient-free simplex (direct search) fitter built on Optim.jl's `NelderMead()`.
Like L-BFGS-B it minimises the scalar sum-of-squared residuals directly, but it
uses no derivative information at all — only function values — which makes it
robust on the non-smooth, multi-lobe BRDF objective at the cost of more
iterations. It is a useful third optimizer for comparing fitting algorithms
(RQ1).

Nelder-Mead in Optim.jl is unconstrained, so box bounds are enforced inside the
objective: parameters are clamped to `[lower, upper]` before evaluation and a
quadratic penalty on the violation pulls the simplex back inside the box.

Sampling, the residual transform, per-model parameter packing, and multi-start
initialisation are shared with the other optimizers via `_fit_common.jl`.

# Usage
```julia
data  = load_brdf("path/to/material.binary")
model = fit_brdf(NelderMeadOpt(), GGXModel, data)
```
"""

using Optim
using StaticArrays
using Colors

# ============================================================================
# Optimizer Type
# ============================================================================

"""
    NelderMeadOpt <: BRDFOptimizer

Gradient-free simplex fitter (via Optim.jl `NelderMead()`).

# Fields
- `n_samples`:     Number of (wi, wo) direction pairs to fit against (default 1024).
- `max_iter`:      Maximum iterations (default 2000; simplex search is slow).
- `x_tol`:         Parameter convergence tolerance (default 1e-6).
- `cosine_weight`: Multiply residuals by `cos(θi)` to down-weight grazing noise.
- `loss`:          Loss function (`:log`, `:linear`, `:cuberoot`, `:ngan`,
                   `:huber`, `:logcosh`).
- `sampling`:      Direction sampling strategy (see `_sample_directions`).
- `n_starts`:      Number of initial guesses; keeps the best fit (default 1).
"""
struct NelderMeadOpt <: BRDFOptimizer
    n_samples::Int
    max_iter::Int
    x_tol::Float64
    cosine_weight::Bool
    loss::Symbol
    sampling::Symbol
    n_starts::Int
end

function NelderMeadOpt(;
    n_samples::Integer = 1024,
    max_iter::Integer = 2000,
    x_tol::Real = 1e-6,
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
    NelderMeadOpt(Int(n_samples), Int(max_iter), Float64(x_tol),
                  cosine_weight, loss, sampling, Int(n_starts))
end

# ============================================================================
# Fit Driver
# ============================================================================

"""
    fit_brdf(opt::NelderMeadOpt, ModelType, data) -> BRDFModel

Fit any model with a registered `_fit_spec`. Runs `opt.n_starts` simplex
minimisations of the (box-penalised) summed squared residuals and returns the
best model.
"""
function fit_brdf(opt::NelderMeadOpt,
                  ::Type{M},
                  data::MeasuredBRDF)::M where {M <: BRDFModel}
    spec        = _fit_spec(M)
    wis, wos    = _sample_directions(opt.sampling, opt.n_samples)
    residual_fn = _residual_function(opt.loss)
    targets     = _fit_targets(data, wis, wos, opt.cosine_weight)

    function objective(p)
        # Clamp into the box for evaluation, penalise the violation so the
        # unconstrained simplex is pushed back inside.
        penalty = 0.0
        pc = similar(p)
        @inbounds for j in eachindex(p)
            lo, hi = spec.lower[j], spec.upper[j]
            c = clamp(p[j], lo, hi)
            d = p[j] - c
            penalty += d * d
            pc[j] = c
        end
        preds = _fit_predict(spec.reconstruct(pc), wis, wos, opt.cosine_weight)
        s = 0.0
        @inbounds for i in eachindex(preds)
            r = residual_fn(preds[i], targets[i])
            s += r * r
        end
        return s + 1e6 * penalty
    end

    options = Optim.Options(iterations = opt.max_iter, x_abstol = opt.x_tol)
    starts  = _generate_starts(spec.p0, spec.lower, spec.upper, opt.n_starts)

    best_param = spec.p0
    best_obj   = Inf
    for p0 in starts
        res = optimize(objective, p0, NelderMead(), options)
        if Optim.minimum(res) < best_obj
            best_obj   = Optim.minimum(res)
            # Clamp the result back into the box (penalty may leave it just outside).
            best_param = clamp.(Optim.minimizer(res), spec.lower, spec.upper)
        end
    end
    return spec.reconstruct(best_param)
end

# ============================================================================
# Option declaration
# ============================================================================

optimizer_options(::Type{NelderMeadOpt}) = OptimizerOption[
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
    OptimizerOption(:max_iter, "Max iterations", :int, 2000,
                    Any[], (200, 20000), 100,
                    "Maximum simplex iterations"),
    OptimizerOption(:n_starts, "Restarts", :int, 1,
                    Any[], (1, 16), 1,
                    "Multi-start initial guesses; keeps the best fit"),
]

# ============================================================================
# Registration
# ============================================================================

export NelderMeadOpt
register_optimizer!(NelderMeadOpt)
