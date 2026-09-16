"""
Barrier-constrained L-BFGS BRDF optimizer.

Box-constrained quasi-Newton fitter built on Optim.jl's `Fminbox(LBFGS())`.
`Fminbox` is *not* the L-BFGS-B algorithm: it is a logarithmic-barrier outer
loop that minimises `f(x) + μ·barrier(x, lower, upper)` with a fully
unconstrained inner L-BFGS, shrinking μ each outer stage. The bounds therefore
enter by bending the objective near the box faces rather than by projection.

That barrier bias is exactly why this optimizer is kept alongside `LBFGSB`
(the native projected-gradient solver in `lbfgsb.jl`). An optimizer sweep over
`test/data/setup` found the barrier variant fits metallic materials slightly
better, so both are registered and the two can be compared directly — which is
itself a result for the optimizer-comparison study (RQ1).

Sampling, the residual transform, per-model parameter packing, and multi-start
initialisation are shared with every other optimizer via `_fit_common.jl`.

# Usage
```julia
data  = load_brdf("path/to/material.binary")
model = fit_brdf(LBFGS(), GGXModel, data)
```
"""

using Optim
using StaticArrays
using Colors

# ============================================================================
# Optimizer Type
# ============================================================================

"""
    LBFGS <: BRDFOptimizer

Barrier box-constrained L-BFGS fitter (via Optim.jl `Fminbox(Optim.LBFGS())`).

Shadows `Optim.LBFGS` inside this module, so Optim's own method type must be
written `Optim.LBFGS()` — see `fit_brdf` below.

# Fields
- `n_samples`:     Number of (wi, wo) direction pairs to fit against (default 1024).
- `max_iter`:      Maximum inner iterations per barrier stage (default 200).
- `x_tol`:         Parameter convergence tolerance (default 1e-6).
- `cosine_weight`: Multiply residuals by `cos(θi)` to down-weight grazing noise.
- `loss`:          Loss function (`:log`, `:linear`, `:cuberoot`, `:ngan`,
                   `:huber`, `:logcosh`).
- `sampling`:      Direction sampling strategy (see `_sample_directions`).
- `n_starts`:      Number of initial guesses; keeps the best fit (default 1).
"""
struct LBFGS <: BRDFOptimizer
    n_samples::Int
    max_iter::Int
    x_tol::Float64
    cosine_weight::Bool
    loss::Symbol
    sampling::Symbol
    n_starts::Int
end

function LBFGS(;
    n_samples::Integer = 1024,
    max_iter::Integer = 200,
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
    LBFGS(Int(n_samples), Int(max_iter), Float64(x_tol),
          cosine_weight, loss, sampling, Int(n_starts))
end

# ============================================================================
# Fit Driver
# ============================================================================

"""
    fit_brdf(opt::LBFGS, ModelType, data) -> BRDFModel

Fit any model with a registered `_fit_spec`. Runs `opt.n_starts` barrier
box-constrained L-BFGS minimisations of the summed squared residuals and
returns the best model.
"""
function fit_brdf(opt::LBFGS,
                  ::Type{M},
                  data::MeasuredBRDF)::M where {M <: BRDFModel}
    spec      = _fit_spec(M)
    wis, wos  = _sample_directions(opt.sampling, opt.n_samples)
    targets   = _fit_targets(data, wis, wos, opt.cosine_weight)

    # Grab the residual closure based on the selected loss
    residual_fn = _residual_function(opt.loss)

    function objective(p)
        preds = _fit_predict(spec.reconstruct(p), wis, wos, opt.cosine_weight)
        s = 0.0
        @inbounds for i in eachindex(preds)
            # Calculate pseudo-residual and square it
            r = residual_fn(preds[i], targets[i])
            s += r * r
        end
        return s
    end

    options = Optim.Options(iterations = opt.max_iter, x_abstol = opt.x_tol)
    starts  = _generate_starts(spec.p0, spec.lower, spec.upper, opt.n_starts)

    best_param = spec.p0
    best_obj   = Inf
    for p0 in starts
        x0 = _clamp_interior(p0, spec.lower, spec.upper)
        # `Optim.LBFGS` is qualified: the bare name is this module's struct.
        res = optimize(objective, spec.lower, spec.upper, x0,
                       Fminbox(Optim.LBFGS()), options)
        if Optim.minimum(res) < best_obj
            best_obj   = Optim.minimum(res)
            best_param = Optim.minimizer(res)
        end
    end
    return spec.reconstruct(best_param)
end

"""
    _clamp_interior(p, lower, upper) -> Vector{Float64}

Nudge `p` strictly inside the box `[lower, upper]` (Fminbox rejects starts on the
boundary). Uses a small relative margin.
"""
function _clamp_interior(p::Vector{Float64},
                         lower::Vector{Float64},
                         upper::Vector{Float64})
    out = similar(p)
    for j in eachindex(p)
        lo, hi = lower[j], upper[j]
        margin = 1e-6 * max(hi - lo, 1.0)
        out[j] = clamp(p[j], lo + margin, hi - margin)
    end
    return out
end

# ============================================================================
# Option declaration
# ============================================================================

optimizer_options(::Type{LBFGS}) = OptimizerOption[
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
                    "Residual mapping (:log HDR, :ngan relative, :huber robust)"),
    OptimizerOption(:n_starts, "Restarts", :int, 1,
                    Any[], (1, 16), 1,
                    "Multi-start initial guesses; keeps the best fit"),
]

# ============================================================================
# Registration
# ============================================================================

export LBFGS
register_optimizer!(LBFGS)
