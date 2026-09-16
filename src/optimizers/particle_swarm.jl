"""
Particle Swarm BRDF optimizer.

Gradient-free, global-ish population fitter built on Optim.jl's `ParticleSwarm`.
A swarm of candidate parameter vectors explores the box `[lower, upper]`,
trading per-iteration cost for a much lower chance of getting trapped in a poor
local minimum than the gradient-based fitters — well suited to the non-convex
BRDF objective with overlapping diffuse/specular lobes. It is a useful global
baseline for comparing fitting algorithms (RQ1).

Box bounds are passed directly to `ParticleSwarm` via its `lower`/`upper`
keywords, so the swarm is confined to the valid parameter region natively.

Sampling, the residual transform, per-model parameter packing, and multi-start
initialisation are shared with the other optimizers via `_fit_common.jl`.

# Usage
```julia
data  = load_brdf("path/to/material.binary")
model = fit_brdf(ParticleSwarmOpt(), GGXModel, data)
```
"""

using Optim
using StaticArrays
using Colors

# ============================================================================
# Optimizer Type
# ============================================================================

"""
    ParticleSwarmOpt <: BRDFOptimizer

Global-ish particle swarm fitter (via Optim.jl `ParticleSwarm`).

# Fields
- `n_samples`:     Number of (wi, wo) direction pairs to fit against (default 1024).
- `max_iter`:      Maximum swarm iterations (default 1000).
- `n_particles`:   Swarm size; 0 lets Optim pick a default (default 0).
- `cosine_weight`: Multiply residuals by `cos(θi)` to down-weight grazing noise.
- `loss`:          Loss function (`:log`, `:linear`, `:cuberoot`, `:ngan`,
                   `:huber`, `:logcosh`).
- `sampling`:      Direction sampling strategy (see `_sample_directions`).
- `n_starts`:      Number of independent swarm runs; keeps the best fit (default 1).
"""
struct ParticleSwarmOpt <: BRDFOptimizer
    n_samples::Int
    max_iter::Int
    n_particles::Int
    cosine_weight::Bool
    loss::Symbol
    sampling::Symbol
    n_starts::Int
end

function ParticleSwarmOpt(;
    n_samples::Integer = 1024,
    max_iter::Integer = 1000,
    n_particles::Integer = 0,
    cosine_weight::Bool = true,
    loss::Symbol = :log,
    sampling::Symbol = :uniform_cosine,
    n_starts::Integer = 1,
)
    loss in (:log, :linear, :cuberoot, :ngan, :huber, :logcosh) ||
        throw(ArgumentError("loss must be :log, :linear, :cuberoot, :ngan, :huber, or :logcosh"))
    sampling in VALID_SAMPLINGS ||
        throw(ArgumentError("sampling must be one of $(VALID_SAMPLINGS); got $sampling"))
    n_particles >= 0 || throw(ArgumentError("n_particles must be >= 0, got $n_particles"))
    n_starts >= 1 || throw(ArgumentError("n_starts must be >= 1, got $n_starts"))
    ParticleSwarmOpt(Int(n_samples), Int(max_iter), Int(n_particles),
                     cosine_weight, loss, sampling, Int(n_starts))
end

# ============================================================================
# Fit Driver
# ============================================================================

"""
    fit_brdf(opt::ParticleSwarmOpt, ModelType, data) -> BRDFModel

Fit any model with a registered `_fit_spec`. Runs `opt.n_starts` particle-swarm
minimisations of the summed squared residuals over the bounded parameter box and
returns the best model.
"""
function fit_brdf(opt::ParticleSwarmOpt,
                  ::Type{M},
                  data::MeasuredBRDF)::M where {M <: BRDFModel}
    spec        = _fit_spec(M)
    wis, wos    = _sample_directions(opt.sampling, opt.n_samples)
    residual_fn = _residual_function(opt.loss)
    targets     = _fit_targets(data, wis, wos, opt.cosine_weight)

    function objective(p)
        preds = _fit_predict(spec.reconstruct(p), wis, wos, opt.cosine_weight)
        s = 0.0
        @inbounds for i in eachindex(preds)
            r = residual_fn(preds[i], targets[i])
            s += r * r
        end
        return s
    end

    method  = ParticleSwarm(lower = spec.lower, upper = spec.upper,
                            n_particles = opt.n_particles)
    options = Optim.Options(iterations = opt.max_iter)
    starts  = _generate_starts(spec.p0, spec.lower, spec.upper, opt.n_starts)

    best_param = spec.p0
    best_obj   = Inf
    for p0 in starts
        x0  = clamp.(p0, spec.lower, spec.upper)
        res = optimize(objective, x0, method, options)
        if Optim.minimum(res) < best_obj
            best_obj   = Optim.minimum(res)
            best_param = clamp.(Optim.minimizer(res), spec.lower, spec.upper)
        end
    end
    return spec.reconstruct(best_param)
end

# ============================================================================
# Option declaration
# ============================================================================

optimizer_options(::Type{ParticleSwarmOpt}) = OptimizerOption[
    OptimizerOption(:sampling, "Sampling", :choice, :uniform_cosine,
                    Any[VALID_SAMPLINGS...], (0, 0), nothing,
                    "Direction sampling strategy"),
    OptimizerOption(:n_samples, "Samples", :int, 1024,
                    Any[], (256, 8192), 256,
                    "Number of (wi, wo) sample pairs"),
    OptimizerOption(:n_particles, "Particles", :int, 0,
                    Any[], (0, 200), 10,
                    "Swarm size (0 = Optim default)"),
    OptimizerOption(:cosine_weight, "Cosine weight", :bool, true,
                    Any[], (0, 0), nothing,
                    "Down-weight grazing-angle residuals"),
    OptimizerOption(:loss, "Loss", :choice, :log,
                    Any[:log, :linear, :cuberoot, :ngan, :huber, :logcosh], (0, 0), nothing,
                    "Loss function (:log HDR, :ngan relative, :huber robust)"),
    OptimizerOption(:max_iter, "Max iterations", :int, 1000,
                    Any[], (100, 10000), 100,
                    "Maximum swarm iterations"),
    OptimizerOption(:n_starts, "Restarts", :int, 1,
                    Any[], (1, 16), 1,
                    "Independent swarm runs; keeps the best fit"),
]

# ============================================================================
# Registration
# ============================================================================

export ParticleSwarmOpt
register_optimizer!(ParticleSwarmOpt)
