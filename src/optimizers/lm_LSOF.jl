"""
    LeastSquaresOptim BRDF optimizer.

Fits analytical BRDF models to measured data by minimising the sum-of-squared
residuals between model predictions and measured values over a fixed set of
sampling directions.

Uses LeastSquaresOptim.jl's `optimize` with a Levenberg-Marquardt routine backed
by stable QR/Cholesky factorizations to jointly estimate all free parameters.
"""

using LeastSquaresOptim
using StaticArrays
using Colors
using LinearAlgebra

# ============================================================================
# Optimizer Type
# ============================================================================

"""
    LeastSquaresOptimFitter <: BRDFOptimizer

Advanced Levenberg-Marquardt or Dogleg fitter (via LeastSquaresOptim.jl).

# Fields
- `n_samples`:     Number of (wi, wo) direction pairs to fit against (default 1024).
- `max_iter`:      Maximum optimization iterations (default 200).
- `x_tol`:         Parameter convergence tolerance (default 1e-6).
- `f_tol`:         Objective change convergence tolerance (default 1e-8).
- `g_tol`:         Gradient convergence tolerance (default 1e-8).
- `cosine_weight`: If `true`, multiply residuals by `cos(θi)` to down-weight
                    grazing-angle noise (default `true`).
- `loss`:          Loss function (`:log`, `:linear`, `:cuberoot`, `:ngan`,
                    `:huber`, `:logcosh`).
- `sampling`:      Direction sampling strategy (`:rusinkiewicz`, `:uniform_half`,
                    `:mixed`, `:uniform`).
- `n_starts`:      Number of initial guesses to try (default 16).
- `solver`:        The internal linear solver, either `:qr` or `:cholesky`.
"""
struct LeastSquaresOptimFitter <: BRDFOptimizer
    n_samples::Int
    max_iter::Int
    x_tol::Float64
    f_tol::Float64
    g_tol::Float64
    cosine_weight::Bool
    loss::Symbol
    sampling::Symbol
    n_starts::Int
    solver::Symbol
end

function LeastSquaresOptimFitter(;
    n_samples::Integer = 1024,
    max_iter::Integer = 200,
    x_tol::Real = 1e-6,
    f_tol::Real = 1e-8,
    g_tol::Real = 1e-8,
    cosine_weight::Bool = true,
    loss::Symbol = :log,
    sampling::Symbol = :uniform_cosine,
    n_starts::Integer = 16,
    solver::Symbol = :qr
)
    loss in (:log, :linear, :cuberoot, :ngan, :huber, :logcosh) ||
        throw(ArgumentError("loss must be :log, :linear, :cuberoot, :ngan, :huber, or :logcosh"))
    sampling in VALID_SAMPLINGS ||
        throw(ArgumentError("sampling must be one of $(VALID_SAMPLINGS); got $sampling"))
    n_starts >= 1 || throw(ArgumentError("n_starts must be >= 1, got $n_starts"))
    solver in (:qr, :cholesky) || throw(ArgumentError("solver must be :qr or :cholesky, got $solver"))

    LeastSquaresOptimFitter(
        Int(n_samples), Int(max_iter), Float64(x_tol), Float64(f_tol),
        Float64(g_tol), cosine_weight, loss, sampling, Int(n_starts), solver
    )
end

# ============================================================================
# Fit Driver
# ============================================================================

"""
    fit_brdf(opt::LeastSquaresOptimFitter, ModelType, data) -> BRDFModel
"""
function fit_brdf(opt::LeastSquaresOptimFitter,
                  ::Type{M},
                  data::MeasuredBRDF)::M where {M <: BRDFModel}
    spec        = _fit_spec(M)
    wis, wos    = _sample_directions(opt.sampling, opt.n_samples)
    loss_fn     = _residual_function(opt.loss)
    targets     = _fit_targets(data, wis, wos, opt.cosine_weight)

    # LeastSquaresOptim minimises sum of squares of the returned residual vector;
    # we return the per-sample pseudo-residual so the loss is applied correctly.
    residual_fn(p) =
        loss_fn.(_fit_predict(spec.reconstruct(p), wis, wos, opt.cosine_weight), targets)

    # Determine backend factorization routine
    backend_solver = opt.solver == :cholesky ? LeastSquaresOptim.Cholesky() : LeastSquaresOptim.QR()
    algo = LeastSquaresOptim.LevenbergMarquardt(backend_solver)

    starts     = _generate_starts(spec.p0, spec.lower, spec.upper, opt.n_starts)
    best_param = spec.p0
    best_obj   = Inf

    for p0 in starts
        result = LeastSquaresOptim.optimize(
            residual_fn, p0, algo;
            lower = spec.lower,
            upper = spec.upper,
            iterations = opt.max_iter,
            x_tol = opt.x_tol,
            f_tol = opt.f_tol,
            g_tol = opt.g_tol,
            autodiff = :forward
        )
        
        obj = result.ssr
        if obj < best_obj
            best_obj   = obj
            best_param = result.minimizer
        end
    end
    return spec.reconstruct(best_param)
end

# ============================================================================
# Option declaration
# ============================================================================

optimizer_options(::Type{LeastSquaresOptimFitter}) = OptimizerOption[
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
    OptimizerOption(:n_starts, "Restarts", :int, 16,
                    Any[], (1, 16), 1,
                    "Multi-start initial guesses; keeps the best fit"),
    OptimizerOption(:solver, "Linear Solver", :choice, :qr,
                    Any[:qr, :cholesky], (0, 0), nothing,
                    "Numerical factorization matrix engine"),
]

# ============================================================================
# Registration
# ============================================================================

export LeastSquaresOptimFitter, fit_brdf
register_optimizer!(LeastSquaresOptimFitter)