"""
L-BFGS-B BRDF optimizer.

Box-constrained quasi-Newton fitter built on Optim.jl's native `Optim.LBFGSB()`
(Byrd, Lu, Nocedal & Zhu 1995) — a bound-constrained solver in its own right,
not a log-barrier wrapper around an unconstrained method. Where
Levenberg-Marquardt operates on the residual vector, this optimizer minimises
the scalar sum-of-squared residuals directly, with gradients obtained by finite
differences. It is a useful second optimizer for comparing fitting algorithms
(RQ1) and tends to be more robust on the flatter, blended diffuse/specular
objective.

See `lbfgs.jl` for the barrier variant (`Fminbox(LBFGS())`), kept alongside this
one because an optimizer sweep over `test/data/setup` found it fits metallic
materials slightly better. `LBFGS` is the default of the two.

Sampling, the residual transform, per-model parameter packing, and multi-start
initialisation are shared with Levenberg-Marquardt via `_fit_common.jl`.

# Usage
```julia
data  = load_brdf("path/to/material.binary")
model = fit_brdf(LBFGSB(), GGXModel, data)
```
"""

using Optim
using StaticArrays
using Colors

# ============================================================================
# Optimizer Type
# ============================================================================

"""
    LBFGSB <: BRDFOptimizer

Box-constrained L-BFGS fitter (via Optim.jl's native `Optim.LBFGSB()`).

# Fields
- `n_samples`:     Number of (wi, wo) direction pairs to fit against (default 1024).
- `max_iter`:      Maximum outer iterations (default 200).
- `x_tol`:         Parameter convergence tolerance (default 1e-6).
- `cosine_weight`: Multiply residuals by `cos(θi)` to down-weight grazing noise.
- `loss`:          Loss function (`:log`, `:linear`, `:cuberoot`, `:ngan`,
                   `:huber`, `:logcosh`).
- `sampling`:      Direction sampling strategy (see `_sample_directions`).
- `n_starts`:      Number of initial guesses; keeps the best fit (default 1).
"""
struct LBFGSB <: BRDFOptimizer
    n_samples::Int
    max_iter::Int
    x_tol::Float64
    cosine_weight::Bool
    loss::Symbol
    sampling::Symbol
    n_starts::Int
end

function LBFGSB(;
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
    LBFGSB(Int(n_samples), Int(max_iter), Float64(x_tol),
           cosine_weight, loss, sampling, Int(n_starts))
end

# ============================================================================
# Fit Driver
# ============================================================================

"""
    fit_brdf(opt::LBFGSB, ModelType, data) -> BRDFModel

Fit any model with a registered `_fit_spec`. Runs `opt.n_starts` box-constrained
L-BFGS minimisations of the summed squared residuals and returns the best model.
"""
function fit_brdf(opt::LBFGSB,
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
        res = optimize(objective, spec.lower, spec.upper, p0,
                       Optim.LBFGSB(), options)
        if Optim.minimum(res) < best_obj
            best_obj   = Optim.minimum(res)
            best_param = Optim.minimizer(res)
        end
    end
    return spec.reconstruct(best_param)
end

# ============================================================================
# Option declaration
# ============================================================================

optimizer_options(::Type{LBFGSB}) = OptimizerOption[
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

export LBFGSB
register_optimizer!(LBFGSB)
