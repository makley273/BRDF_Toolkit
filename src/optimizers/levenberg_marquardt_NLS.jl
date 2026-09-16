"""
    NonlinearSolve SciML BRDF optimizer.

Fits analytical BRDF models to measured data via NonlinearSolve.jl. Utilizes
modern, robust Trust-Region and Levenberg-Marquardt algorithms that offer superior
globalization strategies and numerical handling compared to LsqFit.jl.
"""

using NonlinearSolve
using StaticArrays
using Colors
using LinearAlgebra

# ============================================================================
# Optimizer Type
# ============================================================================

"""
    NonlinearSolveFitter <: BRDFOptimizer

Modern non-linear least squares solver engine leveraging the SciML ecosystem.

# Fields
- `n_samples`:     Number of (wi, wo) direction pairs to fit against (default 1024).
- `max_iter`:      Maximum optimization iterations (default 200).
- `x_tol`:         Step size tolerance (default 1e-6).
- `f_tol`:         Residual minimization tolerance (default 1e-8).
- `cosine_weight`: If `true`, multiply residuals by `cos(θi)` to down-weight
                    grazing-angle noise (default `true`).
- `loss`:          Loss function (`:log`, `:linear`, `:cuberoot`, `:ngan`,
                    `:huber`, `:logcosh`).
- `sampling`:      Direction sampling strategy (`:rusinkiewicz`, `:uniform_half`,
                    `:mixed`, `:uniform`).
- `n_starts`:      Number of initial guesses to try (default 1).
- `algorithm`:     Solver scheme choice (`:trust_region`, `:levenberg_marquardt`).
"""
struct NonlinearSolveFitter <: BRDFOptimizer
    n_samples::Int
    max_iter::Int
    x_tol::Float64
    f_tol::Float64
    cosine_weight::Bool
    loss::Symbol
    sampling::Symbol
    n_starts::Int
    algorithm::Symbol
end

function NonlinearSolveFitter(;
    n_samples::Integer = 1024,
    max_iter::Integer = 200,
    x_tol::Real = 1e-6,
    f_tol::Real = 1e-8,
    cosine_weight::Bool = true,
    loss::Symbol = :log,
    sampling::Symbol = :uniform_cosine,
    n_starts::Integer = 1,
    algorithm::Symbol = :trust_region
)
    loss in (:log, :linear, :cuberoot, :ngan, :huber, :logcosh) ||
        throw(ArgumentError("loss must be :log, :linear, :cuberoot, :ngan, :huber, or :logcosh"))
    sampling in VALID_SAMPLINGS ||
        throw(ArgumentError("sampling must be one of $(VALID_SAMPLINGS); got $sampling"))
    n_starts >= 1 || throw(ArgumentError("n_starts must be >= 1, got $n_starts"))
    algorithm in (:trust_region, :levenberg_marquardt) ||
        throw(ArgumentError("algorithm must be :trust_region or :levenberg_marquardt, got $algorithm"))

    NonlinearSolveFitter(
        Int(n_samples), Int(max_iter), Float64(x_tol), Float64(f_tol),
        cosine_weight, loss, sampling, Int(n_starts), algorithm
    )
end

# ============================================================================
# Fit Driver
# ============================================================================

"""
    fit_brdf(opt::NonlinearSolveFitter, ModelType, data) -> BRDFModel
"""
function fit_brdf(opt::NonlinearSolveFitter,
                  ::Type{M},
                  data::MeasuredBRDF)::M where {M <: BRDFModel}
    spec        = _fit_spec(M)
    wis, wos    = _sample_directions(opt.sampling, opt.n_samples)
    residual_fn = _residual_function(opt.loss)
    targets     = _fit_targets(data, wis, wos, opt.cosine_weight)

    # NonlinearSolve.jl handles vector-to-vector least-squares targets directly:
    # f(p, u) is driven to zero, so we return the per-sample pseudo-residual.
    nonlinear_residual_fn(p, _) =
        residual_fn.(_fit_predict(spec.reconstruct(p), wis, wos, opt.cosine_weight), targets)

    # Map selected algorithm string to explicit SciML implementations
    solver_algo = if opt.algorithm == :trust_region
        NonlinearSolve.TrustRegion()
    else
        NonlinearSolve.LevenbergMarquardt()
    end

    starts     = _generate_starts(spec.p0, spec.lower, spec.upper, opt.n_starts)
    best_param = spec.p0
    best_obj   = Inf

    for p0 in starts
        # Construct explicit least squares problem definition with box constraints
        prob = NonlinearSolve.NonlinearLeastSquaresProblem(
            nonlinear_residual_fn, p0;
            lb = spec.lower,
            ub = spec.upper
        )
        
        result = NonlinearSolve.solve(
            prob, solver_algo;
            maxiters = opt.max_iter,
            abstol = opt.f_tol,
            reltol = opt.x_tol
        )
        
        # Calculate objective criteria manually via sum of squares of residuals
        resid = nonlinear_residual_fn(result.u, nothing)
        obj   = sum(abs2, resid)
        
        if obj < best_obj
            best_obj   = obj
            best_param = result.u
        end
    end
    return spec.reconstruct(best_param)
end

# ============================================================================
# Option declaration
# ============================================================================

optimizer_options(::Type{NonlinearSolveFitter}) = OptimizerOption[
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
    OptimizerOption(:algorithm, "Algorithm", :choice, :trust_region,
                    Any[:trust_region, :levenberg_marquardt], (0, 0), nothing,
                    "SciML underlying optimizer backend engine"),
]

# ============================================================================
# Registration
# ============================================================================

export NonlinearSolveFitter, fit_brdf
register_optimizer!(NonlinearSolveFitter)