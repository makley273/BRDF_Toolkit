"""
Shared fitting infrastructure used by all BRDF optimizers.

This file is loaded before the optimizer plugins (filename sorts ahead
of them) and defines the pieces every fitter needs, so that adding a new
optimizer only requires implementing its solver loop — not re-deriving sampling,
the residual transform, parameter packing, or multi-start initialisation.

Contents:
- Direction sampling (`_sample_directions`) shared across optimizers.
- The loss function (`_residual_function`): `:linear`, `:log`, `:cuberoot`,
  `:ngan`, `:huber`, `:logcosh`.
- Target/prediction helpers and the multi-start generator.

Per-model fit specs (`_fit_spec`) live with each model in `src/models/` — the
shared bounds and microfacet factories are in `src/models/_fit_spec.jl`.
"""

using StaticArrays
using Colors
using LinearAlgebra

# ============================================================================
# Low-discrepancy sampling
# ============================================================================

"""
    _halton(i, base) -> Float64

Radical-inverse Halton sample (low-discrepancy 1D quasi-random).
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
    _sample_directions_uniform_cosine(n) -> (wis, wos)

Independent cosine-weighted hemisphere sampling of wi and wo. Each direction is
drawn from its own pair of Halton dimensions (bases 2,3 for wi; 5,7 for wo) so
wi and wo are genuinely independent — for any wi the outgoing direction ranges
over the whole hemisphere. Broad coverage makes this the diagnostic baseline and
the broad-coverage majority of `:mixed`; it under-samples sharp specular peaks
relative to the half-angle strategies.
"""
function _sample_directions_uniform_cosine(n::Int)
    wis = Vector{SVector{3, Float32}}(undef, n)
    wos = Vector{SVector{3, Float32}}(undef, n)

    for k in 1:n
        u1 = _halton(k, 2); u2 = _halton(k, 3)   # wi: bases 2,3
        u3 = _halton(k, 5); u4 = _halton(k, 7)   # wo: bases 5,7 (independent)

        r = sqrt(u1); phi_i = 2π * u2
        wis[k] = SVector{3, Float32}(r * cos(phi_i), r * sin(phi_i),
                                     sqrt(max(0.0, 1.0 - u1)))

        r2 = sqrt(u3); phi_o = 2π * u4
        wos[k] = SVector{3, Float32}(r2 * cos(phi_o), r2 * sin(phi_o),
                                     sqrt(max(0.0, 1.0 - u3)))
    end
    return wis, wos
end

"""
    _sample_directions_uniform(n) -> (wis, wos)

Independent **uniform solid-angle** sampling of wi and wo — the truly unbiased
baseline. Unlike `_sample_directions_uniform_cosine` (cosine-weighted, which crowds the
normal), every equal solid-angle patch of the hemisphere gets the same expected
number of samples: `cos θ` is drawn uniformly on `(0, 1]` (`z = u`,
`r = √(1−z²)`), `φ` uniform on `[0, 2π)`. wi and wo use disjoint Halton
dimensions (bases 2,3 for wi; 5,7 for wo) so they are genuinely independent and
for any wi the wo cloud spans the whole hemisphere with flat density.

This carries *no* physical importance weighting at all — it does not favour the
specular ridge, retroreflection, or the normal. That makes it the honest
reference for measuring how much the biased strategies help (or a good choice
when the fit metric already supplies its own weighting).
"""
function _sample_directions_uniform(n::Int)
    wis = Vector{SVector{3, Float32}}(undef, n)
    wos = Vector{SVector{3, Float32}}(undef, n)

    for k in 1:n
        u1 = _halton(k, 2); u2 = _halton(k, 3)   # wi: bases 2,3
        u3 = _halton(k, 5); u4 = _halton(k, 7)   # wo: bases 5,7 (independent)

        z1 = u1; r1 = sqrt(max(0.0, 1.0 - z1 * z1)); phi1 = 2π * u2
        wis[k] = SVector{3, Float32}(r1 * cos(phi1), r1 * sin(phi1), z1)

        z2 = u3; r2 = sqrt(max(0.0, 1.0 - z2 * z2)); phi2 = 2π * u4
        wos[k] = SVector{3, Float32}(r2 * cos(phi2), r2 * sin(phi2), z2)
    end
    return wis, wos
end

"""
    _sample_directions_halfangle(n; theta_h_power=2, theta_h_max=π/2, idx_offset=0) -> (wis, wos)

Sample `(wi, wo)` pairs in Rusinkiewicz half-angle space `(θ_h, φ_h, θ_d, φ_d)`.

`θ_h = 0` corresponds to the perfect mirror direction. `theta_h_power` controls
how strongly samples are biased toward the specular ridge:
- `2` (default) — quadratic remap matching MERL's sqrt index mapping. Best for
  sharp isotropic specular lobes.
- `1` — uniform in θ_h. Still hits the ridge (parameterization is centered on
  it) but spreads samples across the full lobe — better for anisotropic shapes
  and for keeping the F0 Jacobian column well-conditioned without over-biasing.

`theta_h_max` caps θ_h; pass a small value to draw *only* near-specular pairs
(used by `:mixed` sampling to top up F0 coverage).

`idx_offset` shifts the Halton stream so callers can mix multiple halfangle
draws without overlap.

Invalid samples (wi or wo below the horizon) are replaced with the next valid
Halton draw.
"""
function _sample_directions_halfangle(n::Int;
                                       theta_h_power::Real = 2,
                                       theta_h_max::Real = π/2,
                                       idx_offset::Int = 0)
    wis = Vector{SVector{3, Float32}}(undef, n)
    wos = Vector{SVector{3, Float32}}(undef, n)

    N = SVector{3, Float32}(0.0f0, 0.0f0, 1.0f0)
    k = 1
    idx = 0
    max_tries = 50 * n
    while k <= n && idx < max_tries
        idx += 1
        u1 = _halton(idx + idx_offset, 2)
        u2 = _halton(idx + idx_offset, 3)
        u3 = _halton(idx + idx_offset, 5)
        u4 = _halton(idx + idx_offset, 7)

        # Power remap controls θ_h bias toward the specular ridge
        theta_h = Float32((u1 ^ theta_h_power) * theta_h_max)
        phi_h   = Float32(2π * u2)
        theta_d = Float32(u3 * (π / 2))
        # Full φ_d ∈ [0, 2π): sample the whole outgoing hemisphere. The [0, π)
        # fold (φ_d ↔ φ_d+π is the wi↔wo swap) is only valid for a *reciprocal*
        # target; keeping it here silently halved the input domain and, for a
        # non-reciprocity-invariant model parameterization, would leave the other
        # half untrained. The reciprocity fold belongs solely in the MERL storage
        # loader (`map_phi_d_continuous`), not in the shared direction generator.
        phi_d   = Float32(2π * u4)

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
    _sample_directions_mixed(n) -> (wis, wos)

Combine independent uniform `(wi, wo)` sampling with a smaller batch of
near-specular half-angle draws. The uniform majority preserves broad off-peak
coverage (important for anisotropic lobe shape), while the near-specular minority
ensures F0 has a non-degenerate Jacobian column.

Split: 80% uniform + 20% near-specular (θ_h ∈ [0, π/8], uniform).
"""
function _sample_directions_mixed(n::Int)
    n_spec = max(1, n ÷ 5)
    n_uni  = n - n_spec

    wis_u, wos_u = _sample_directions_uniform_cosine(n_uni)
    wis_s, wos_s = _sample_directions_halfangle(n_spec;
                                                 theta_h_power = 1,
                                                 theta_h_max   = π/8)
    return vcat(wis_u, wis_s), vcat(wos_u, wos_s)
end

"""
    _sample_directions_fibonacci(n) -> (wis, wos)

Deterministic **Fibonacci (golden-spiral) lattice**, taken as a *full product* over
the joint `(wi, wo)` space. A single even hemisphere lattice `M` of `m = round(√n)`
points is built analytically — `cos θ` steps uniformly through `(0, 1]` (equal
solid angle), azimuth advances by the golden angle `π(3 − √5)` — and every incoming
lattice direction is paired with every outgoing one: `{(x, y) : x, y ∈ M}`. That
grid tiles the whole `wi × wo` domain with the most even per-hemisphere coverage
available (lower discrepancy than Halton, no clumps), which is why the product beats
the earlier strided-permutation pairing: the permutation only walked a diagonal
slice of the joint space, whereas this covers it completely.

Returns exactly `m² ≈ n` pairs (`n` is rounded to the nearest perfect square so the
product stays complete). The `m` diagonal pairs with `x = y` are the exact
retroreflection configurations; both orderings `(x, y)` and `(y, x)` are present.
Every caller sizes its work off `length(wis)`, so the `m² ≠ n` count is harmless.
"""
function _sample_directions_fibonacci(n::Int)
    m = max(1, round(Int, sqrt(n)))               # per-hemisphere lattice size; m² ≈ n
    golden_angle = π * (3.0 - sqrt(5.0))          # ≈ 2.399963 rad

    # One even hemisphere lattice (equal-solid-angle in z, golden-angle azimuth).
    F = Vector{SVector{3, Float32}}(undef, m)
    for i in 1:m
        z = 1.0 - (i - 0.5) / m                    # cos θ ∈ (0, 1], uniform → equal area
        r = sqrt(max(0.0, 1.0 - z * z))
        a = (i - 1) * golden_angle
        F[i] = SVector{3, Float32}(r * cos(a), r * sin(a), z)
    end

    # Full Cartesian product M × M: every wi lattice dir against every wo lattice dir.
    npairs = m * m
    wis = Vector{SVector{3, Float32}}(undef, npairs)
    wos = Vector{SVector{3, Float32}}(undef, npairs)
    k = 1
    @inbounds for i in 1:m, j in 1:m
        wis[k] = F[i]
        wos[k] = F[j]
        k += 1
    end
    return wis, wos
end

"""
    _sample_directions(sampling::Symbol, n::Int) -> (wis, wos)

Dispatch a direction-sampling strategy. Shared by every optimizer so that the
`:sampling` option behaves identically regardless of the solver.

Strategies:
- `:rusinkiewicz`  — half-angle, quadratic θ_h bias toward the specular ridge.
- `:uniform_half`  — half-angle, uniform θ_h.
- `:mixed`         — 80% uniform + 20% near-specular.
- `:uniform`         — independent uniform solid-angle `(wi, wo)`; truly unbiased baseline.
- `:uniform_cosine`  — independent cosine-weighted `(wi, wo)`; the default (beat `:rusinkiewicz` on 294/312 materials by SSIM).
- `:fibonacci`     — golden-spiral lattice √n×√n full product; complete even (wi,wo) grid.

`VALID_SAMPLINGS` lists these symbols for option validation across optimizers.
"""
function _sample_directions(sampling::Symbol, n::Int)
    if sampling === :rusinkiewicz
        return _sample_directions_halfangle(n; theta_h_power = 2)
    elseif sampling === :uniform_half
        return _sample_directions_halfangle(n; theta_h_power = 1)
    elseif sampling === :mixed
        return _sample_directions_mixed(n)
    elseif sampling === :uniform_cosine
        return _sample_directions_uniform_cosine(n)
    elseif sampling === :uniform
        return _sample_directions_uniform(n)
    elseif sampling === :fibonacci
        return _sample_directions_fibonacci(n)
    else
        throw(ArgumentError("unknown sampling strategy: $sampling"))
    end
end

"""
    VALID_SAMPLINGS

Canonical tuple of accepted `:sampling` symbols, shared by every optimizer's
option validation and UI dropdown so a new strategy is exposed everywhere by
editing this one list (plus the `_sample_directions` dispatch).
"""
const VALID_SAMPLINGS = (:rusinkiewicz, :uniform_half, :mixed, :uniform_cosine,
                         :uniform, :fibonacci)

# ============================================================================
# Residual transform
# ============================================================================

# """
#     _residual_transform(loss::Symbol) -> Function

# Return the per-value transform applied to BRDF magnitudes before computing
# residuals. Compressing the large specular/diffuse dynamic range lets both
# regimes contribute meaningfully to the fit.

# - `:linear`   — identity (raw `f`).
# - `:log`      — `log1p(f)`. Strong compression; matches MERL's HDR range.
# - `:cuberoot` — `cbrt(f)`. Moderate compression; empirically the best match to
#   human perceptual judgement of BRDF approximation quality (RMS-CRWE).
# """
# function _residual_transform(loss::Symbol)
#     if loss === :linear
#         return identity
#     elseif loss === :log
#         return x -> log1p(max(x, 0.0))
#     elseif loss === :cuberoot
#         return x -> cbrt(max(x, 0.0))
#     else
#         throw(ArgumentError("unknown loss: $loss"))
#     end
# end
"""
    _residual_function(loss::Symbol; epsilon=1e-6, delta=0.1, log_eps=1e-3, cbrt_eps=1e-5) -> Function

Returns a function `f(pred, target)` that computes a pseudo-residual.
Optimizers will square this value (R^2), so we return R such that R^2
equals the desired loss penalty.

`epsilon` is the `:ngan` denominator floor; `log_eps` is the `:log` floor;
`cbrt_eps` is the `:cuberoot` floor.
"""
function _residual_function(loss::Symbol; epsilon=1e-6, delta=0.1, log_eps=1e-3, cbrt_eps=1e-5)
    if loss === :linear
        return (p, t) -> p - t
    elseif loss === :log
        # Relative log: R = log((p + ε) / (t + ε)). NOT log1p — for BRDF values
        # (mostly ≪ 1) log1p(x) ≈ x, i.e. a linear loss that leaves the dark
        # hemisphere with vanishing gradients. The specular peak's slope then
        # drags the diffuse floor to 0 unopposed, and the collapse worsens with
        # network width — a fitted neural BRDF rendered black everywhere except
        # the highlight. The ε-floored ratio keeps dark regions logarithmic
        # (real gradients) while still compressing the peak. ε = 1e-3 matches
        # the error-map relative-error floor.
        return (p, t) -> log(max(p, 0.0) + log_eps) - log(max(t, 0.0) + log_eps)
    elseif loss === :cuberoot
        # R = cbrt(p) - cbrt(t). Bare cbrt is finite-VALUED everywhere (the
        # max(·,0) guards negatives), but its DERIVATIVE d/dx cbrt(x) =
        # 1/(3·cbrt(x)²) is +Inf at x=0 and ~3e7 near it. A metal like
        # weta_brushed_steel_satin is ≈0 over most of the hemisphere, so many
        # samples have pred≈0; a single one makes the LM Jacobian / Zygote-Adam
        # gradient Inf → the whole step is NaN (same failure the :logcosh guard
        # below documents). log/linear/ngan survive because their slopes are
        # bounded. Flooring the argument by cbrt_eps — symmetrically on p and t
        # so R still vanishes iff p==t — caps the slope at 1/(3·cbrt_eps^(2/3))
        # (~720 at 1e-5, the same order as :log's 1/log_eps ceiling the user
        # confirms is stable) while shifting the dark floor by only cbrt(1e-5)
        # ≈ 0.02, negligible against typical cube-root magnitudes.
        return (p, t) -> cbrt(max(p, 0.0) + cbrt_eps) - cbrt(max(t, 0.0) + cbrt_eps)
    elseif loss === :ngan
        # Ngan et al. relative error. The optimizer squares R, so the denominator
        # must be sqrt(t + eps) to give R^2 = (p - t)^2 / (t + eps): a relative
        # metric that balances bright and dark regions.
        #
        # NOTE: dividing by (t + eps) directly (so R^2 = (p-t)^2/(t+eps)^2) is
        # wrong — it weights dark samples by 1/eps^2 (~1e6), and since most of
        # the hemisphere is dark the global minimum collapses to "predict 0
        # everywhere", i.e. a black diffuse material. The sqrt fixes that.
        return (p, t) -> (p - t) / sqrt(max(t, 0.0) + epsilon)
    elseif loss === :huber
        # R^2 = Huber(d). We return R = sign(d) * sqrt(Huber(|d|)).
        # Scale caveat: with the default delta=0.1 on RAW BRDF values, the
        # dark hemisphere (|d| ≪ 0.1) sees a plain squared loss (≈ :linear)
        # while the specular peak (values up to ~1e3 on MERL) is entirely in
        # the |d|-damped branch — i.e. the peak is treated as an outlier and
        # under-fit. That is Huber working as designed, not a bug; for HDR
        # reflectance prefer :log/:cuberoot, or raise delta toward the peak
        # magnitude to keep it quadratically weighted.
        return (p, t) -> begin
            d = p - t
            abs_d = abs(d)
            if abs_d <= delta
                return d / sqrt(2.0)  # R^2 = 0.5 * d^2
            else
                return sign(d) * sqrt(delta * (abs_d - 0.5 * delta))
            end
        end
    elseif loss === :logcosh
        # R^2 = log(cosh(d)).
        return (p, t) -> begin
            d = p - t
            abs_d = abs(d)
            # Small |d|: log(cosh d) = d²/2·(1 − d²/6 + …), so R = d/√2 (the
            # relative R² error of dropping the tail is ≤ ~2e-5 at the 1e-2
            # boundary). The guard must be this wide for Float32 inputs (the
            # neural trainer): cosh(d) == 1f0 EXACTLY for |d| ≲ 5e-4 (eps(1f0)
            # ≈ 1.2e-7), so the naive log(cosh(d)) is 0 with d ≠ 0 — the
            # residual loses its gradient and sqrt at 0 has an infinite
            # derivative; a single such sample NaNs the whole Zygote/Adam
            # training step. The old 1e-5 guard was calibrated for Float64.
            if abs_d < 1e-2
                return d / sqrt(2.0)
            end
            # Stable for ALL d (no cosh overflow, no cancellation):
            # log(cosh d) = |d| + log1p(e^{-2|d|}) − log 2.
            lcosh = abs_d + log1p(exp(-2 * abs_d)) - log(2.0)
            return sign(d) * sqrt(lcosh)
        end
    else
        throw(ArgumentError("unknown loss: $loss"))
    end
end

# ============================================================================
# Targets and predictions
# ============================================================================

"""
    _fit_targets(data, wis, wos, cosine_weight) -> Vector{Float64}

Build the (3N) target vector of raw (optionally cosine-weighted) BRDF values from
measured `data` at the sample directions. The loss/transform is applied later,
per residual, by `_residual_function`.
"""
function _fit_targets(data::MeasuredBRDF,
                      wis::Vector{SVector{3, Float32}},
                      wos::Vector{SVector{3, Float32}},
                      cosine_weight::Bool)
    n = length(wis)
    t = Vector{Float64}(undef, 3n)
    for k in 1:n
        c = lookup(data, wis[k], wos[k])
        w = cosine_weight ? Float64(wis[k][3]) : 1.0
        t[3k - 2] = Float64(c.r) * w
        t[3k - 1] = Float64(c.g) * w
        t[3k    ] = Float64(c.b) * w
    end
    return t
end

"""
    _fit_predict(model, wis, wos, cosine_weight) -> Vector{T}

Build the (3N) prediction vector of raw (optionally cosine-weighted) BRDF values
for `model` at the sample directions, applying the same weighting as
`_fit_targets`.  The element type T is
inferred from the first evaluation so that ForwardDiff Dual values propagate
correctly when the optimizer computes the Jacobian.
"""
function _fit_predict(model::BRDFModel,
                      wis::Vector{SVector{3, Float32}},
                      wos::Vector{SVector{3, Float32}},
                      cosine_weight::Bool)
    n  = length(wis)
    c1 = evaluate(model, wis[1], wos[1])
    w1 = cosine_weight ? wis[1][3] : 1.0
    T  = typeof(c1.r * w1) # Let ForwardDiff Duals propagate cleanly
    p  = Vector{T}(undef, 3n)
    for k in 1:n
        c = evaluate(model, wis[k], wos[k])
        w = cosine_weight ? wis[k][3] : 1.0
        p[3k - 2] = c.r * w
        p[3k - 1] = c.g * w
        p[3k    ] = c.b * w
    end
    return p
end

# ============================================================================
# Multi-start initialisation
# ============================================================================

"""
    _generate_starts(p0, lower, upper, n_starts) -> Vector{Vector{Float64}}

Return `n_starts` initial parameter vectors. The first is always the model's
default guess `p0`; the rest are spread across the bounded box using a
deterministic per-dimension Halton sequence (reproducible, no RNG).

Multi-start guards against the Levenberg-Marquardt / L-BFGS-B tendency to settle
into a poor local minimum from a single fixed seed (overlapping diffuse/specular
lobes make the BRDF objective non-convex).
"""
function _generate_starts(p0::Vector{Float64},
                          lower::Vector{Float64},
                          upper::Vector{Float64},
                          n_starts::Int)
    starts = Vector{Vector{Float64}}()
    push!(starts, copy(p0))
    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31)
    for s in 1:(n_starts - 1)
        p = Vector{Float64}(undef, length(p0))
        for j in eachindex(p0)
            b  = primes[mod1(j, length(primes))]
            u  = _halton(s, b)
            lo, hi = lower[j], upper[j]
            p[j] = lo + u * (hi - lo)
        end
        push!(starts, p)
    end
    return starts
end
