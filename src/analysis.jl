"""
Post-fit analysis: BRDF-space and image-space error metrics.

Three groups of functions comparing a fitted `BRDFModel` against `MeasuredBRDF`
data:

1. BRDF-space metrics (pointwise over a direction lattice). All three share one
   body and differ only in the compression applied to the luminance; all three
   are cosine-weighted by default (`cosine_weight=false` turns it off), matching
   the weighting the optimizers apply to the fit residuals:
   - `brdf_rmse`     — linear luminance RMSE (raw magnitude difference).
   - `brdf_rms_crwe` — Root-Mean-Squared Cube-Root-Weighted Error.
   - `brdf_rmsle`    — Root-Mean-Squared Logarithmic Weighted Error. The
     strongest compression, and the headline metric: it correlates best with
     human judgement of BRDF approximation quality among BRDF-space measures.

2. Image-space metric (neighborhood-based, on CPU renders of the sphere):
   - `image_ssim` — SSIM (Wang et al. 2004) between renders of the measured
     data and the fitted model, using `render_brdf_sphere` (which mirrors the
     browser render path: cosine term, intensity gain, Reinhard tone mapping,
     no gamma). Answers "does the fitted material LOOK like the measured one",
     complementing the pointwise BRDF-space metrics.

3. Physical sanity:
   - `reciprocity_error` — checks Helmholtz reciprocity f(wi,wo) == f(wo,wi).
"""

using StaticArrays
using Colors

# ============================================================================
# Hemisphere sampling (uniform solid angle)
# ============================================================================

"""
    _uniform_hemisphere(n) -> Vector{SVector{3,Float32}}

`n` directions uniformly distributed over the upper hemisphere by solid angle
(golden-ratio quasi-random; pdf = 1/2π). Deterministic for reproducibility.
"""
function _uniform_hemisphere(n::Int)
    dirs = Vector{SVector{3, Float32}}(undef, n)
    g1 = 0.7548776662466927    # 1 / φ
    g2 = 0.5698402909980532    # 1 / φ²
    for k in 1:n
        u1 = mod(k * g1, 1.0)          # cosθ uniform → uniform solid angle
        u2 = mod(k * g2, 1.0)
        z  = u1
        r  = sqrt(max(0.0, 1.0 - z * z))
        phi = 2π * u2
        dirs[k] = SVector{3, Float32}(r * cos(phi), r * sin(phi), z)
    end
    return dirs
end

# ============================================================================
# BRDF-space error metrics
# ============================================================================

@inline _luminance(c) = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b

"""
    _metric_directions(n) -> (wis, wos)

Deterministic Fibonacci-lattice (wi, wo) pairs for diagnostic error metrics.
Matches the lattice historically used by the UI's RMSE readout.
"""
function _metric_directions(n::Int)
    wis = Vector{SVector{3, Float32}}(undef, n)
    wos = Vector{SVector{3, Float32}}(undef, n)
    g1 = 0.7548776662466927
    g2 = 0.5698402909980532
    for k in 1:n
        u1 = mod(k * g1, 1.0); u2 = mod(k * g2, 1.0)
        v1 = mod((k + 0.5) * g1 + 0.3, 1.0); v2 = mod((k + 0.5) * g2 + 0.7, 1.0)
        r  = sqrt(u1);  phi_i = 2π * u2
        r2 = sqrt(v1);  phi_o = 2π * v2
        wis[k] = SVector{3, Float32}(r * cos(phi_i), r * sin(phi_i),  sqrt(max(0.0, 1.0 - u1)))
        wos[k] = SVector{3, Float32}(r2 * cos(phi_o), r2 * sin(phi_o), sqrt(max(0.0, 1.0 - v1)))
    end
    return wis, wos
end

"""
    _metric_rms(measured, model, transform, n, cosine_weight) -> Float64

Shared body of the three BRDF-space metrics: root mean square of
`transform(L_meas) - transform(L_model)` over the deterministic lattice, where
`L` is the luminance optionally multiplied by `cos(theta_i)`.

The cosine is applied to the raw luminance BEFORE the transform, matching
`_fit_targets`/`_fit_predict` in the optimizers (which weight the raw BRDF and
let `_residual_function` compress afterwards). Metric and fit objective
therefore see the same quantity.
"""
function _metric_rms(measured::MeasuredBRDF, model::BRDFModel,
                     transform::F, n::Int, cosine_weight::Bool) where {F}
    wis, wos = _metric_directions(n)
    s = 0.0
    for k in 1:n
        w  = cosine_weight ? Float64(wis[k][3]) : 1.0
        lm = Float64(_luminance(lookup(measured, wis[k], wos[k]))) * w
        lf = Float64(_luminance(evaluate(model, wis[k], wos[k]))) * w
        s += (transform(lm) - transform(lf))^2
    end
    return sqrt(s / max(n, 1))
end

"""
    brdf_rmse(measured, model; n=1024, cosine_weight=true) -> Float64

Linear luminance RMSE between measured data and the fitted model over a
deterministic (wi, wo) lattice. Sensitive to the specular peak's large
magnitude — included for comparison against the compressed metrics.
"""
brdf_rmse(measured::MeasuredBRDF, model::BRDFModel;
          n::Int = 1024, cosine_weight::Bool = true) =
    _metric_rms(measured, model, identity, n, cosine_weight)

"""
    brdf_rms_crwe(measured, model; n=1024, cosine_weight=true) -> Float64

Root-Mean-Squared Cube-Root-Weighted Error: RMSE on cube-root-compressed,
cosine-weighted luminance. The cube root mirrors the non-linear luminosity
response of the eye; the `cos(theta_i)` factor is what the "W" in CRWE denotes
in the literature (Forés et al.), so disabling `cosine_weight` yields the
unweighted RMS-CRE instead.
"""
brdf_rms_crwe(measured::MeasuredBRDF, model::BRDFModel;
              n::Int = 1024, cosine_weight::Bool = true) =
    _metric_rms(measured, model, x -> cbrt(max(x, 0.0)), n, cosine_weight)

"""
    brdf_rmsle(measured, model; n=1024, cosine_weight=true) -> Float64

Root-Mean-Squared Logarithmic Weighted Error (RMS-LogWE): RMSE on
`log1p`-compressed, cosine-weighted luminance. The logarithm compresses more
strongly than the cube root of `brdf_rms_crwe`, so this metric weights the
low-magnitude diffuse lobe more heavily and the specular peak less. It is the
headline metric for model selection: of the BRDF-space measures it correlates
best with human judgement of BRDF approximation quality (Kavoosighafi et al.
report Spearman 0.52 for RMS-LogWE against 0.40 for RMS-CRWE). Negative
luminances (measurement noise) are clamped to 0 so `log1p` stays real.
Disabling `cosine_weight` yields the unweighted RMS-LogE.
"""
brdf_rmsle(measured::MeasuredBRDF, model::BRDFModel;
           n::Int = 1024, cosine_weight::Bool = true) =
    _metric_rms(measured, model, x -> log1p(max(x, 0.0)), n, cosine_weight)

# ============================================================================
# Image-space metric: SSIM on CPU sphere renders
# ============================================================================

# Gaussian window taps (normalized). Standard SSIM window: 11×11, σ = 1.5.
function _gauss_kernel(radius::Int, sigma::Float64)
    k = [exp(-(i^2) / (2 * sigma^2)) for i in -radius:radius]
    return k ./ sum(k)
end

# Separable Gaussian filter with clamp-to-edge padding.
function _gauss_filter(img::Matrix{Float64}, k::Vector{Float64})
    r = (length(k) - 1) ÷ 2
    h, w = size(img)
    tmp = similar(img)
    out = similar(img)
    @inbounds for i in 1:h, j in 1:w
        s = 0.0
        for d in -r:r
            s += k[d + r + 1] * img[i, clamp(j + d, 1, w)]
        end
        tmp[i, j] = s
    end
    @inbounds for i in 1:h, j in 1:w
        s = 0.0
        for d in -r:r
            s += k[d + r + 1] * tmp[clamp(i + d, 1, h), j]
        end
        out[i, j] = s
    end
    return out
end

"""
    _ssim_map(a, b) -> Matrix{Float64}

Per-pixel SSIM map between two luminance images in [0,1] (Wang et al. 2004:
11×11 Gaussian window σ=1.5, K1=0.01, K2=0.03, dynamic range L=1 — valid here
because inputs are Reinhard-tone-mapped into [0,1)).
"""
function _ssim_map(a::Matrix{Float64}, b::Matrix{Float64})
    k = _gauss_kernel(5, 1.5)
    mu_a = _gauss_filter(a, k)
    mu_b = _gauss_filter(b, k)
    var_a  = _gauss_filter(a .* a, k) .- mu_a .^ 2
    var_b  = _gauss_filter(b .* b, k) .- mu_b .^ 2
    cov_ab = _gauss_filter(a .* b, k) .- mu_a .* mu_b
    C1 = 0.01^2
    C2 = 0.03^2
    return @. ((2 * mu_a * mu_b + C1) * (2 * cov_ab + C2)) /
              ((mu_a^2 + mu_b^2 + C1) * (var_a + var_b + C2))
end

_lum_matrix(img::Matrix{RGB{Float32}}) = Float64[_luminance(c) for c in img]

# Default light directions for `image_ssim`: three incidence angles from
# near-normal to near-grazing, with varied azimuth (matters for anisotropy).
# (θ, φ) in the app's light convention (θ=0 → light on the camera axis).
const SSIM_LIGHT_DIRS = [
    (Float32(π / 8), 0.0f0),
    (Float32(π / 4), Float32(π / 3)),
    (Float32(3π / 8), Float32(2π / 3)),
]

"""
    image_ssim(measured, model;
               light_dirs=SSIM_LIGHT_DIRS, resolution=256, intensity=π)
        -> (mean, min, per_light)

Image-space fit quality: render the measured data and the fitted model with the
CPU twin of the browser pipeline (`render_brdf_sphere`) under each light
direction, compute SSIM on the luminance images, and mask the mean to sphere
pixels (the shared constant background would otherwise inflate the score;
windows straddling the silhouette still see identical background in both
images, which is a small, documented bias toward 1).

Returns per-light SSIM values plus their mean (headline number) and min
(worst-case lighting). SSIM = 1 means visually identical renders.
"""
function image_ssim(measured::MeasuredBRDF, model::BRDFModel;
                    light_dirs = SSIM_LIGHT_DIRS,
                    resolution::Int = 256, intensity::Real = Float64(π))
    vals = Float64[]
    for (th, ph) in light_dirs
        img_m, mask = render_brdf_sphere(measured, th, ph;
                                         resolution = resolution, intensity = intensity)
        img_f, _    = render_brdf_sphere(model, th, ph;
                                         resolution = resolution, intensity = intensity)
        smap = _ssim_map(_lum_matrix(img_m), _lum_matrix(img_f))
        s = 0.0; n = 0
        @inbounds for idx in eachindex(mask)
            if mask[idx]
                s += smap[idx]; n += 1
            end
        end
        push!(vals, n > 0 ? s / n : 1.0)
    end
    return (mean = sum(vals) / length(vals), min = minimum(vals), per_light = vals)
end

# ============================================================================
# Physical sanity: Helmholtz reciprocity
# ============================================================================

"""
    reciprocity_error(model; n_pairs=2000, rtol=1e-3) -> (max_abs_diff, verdict)

Check Helmholtz reciprocity by comparing `f(wi, wo)` with `f(wo, wi)` over random
direction pairs. Returns the maximum absolute per-channel difference and a
verdict (`:ok` if the relative difference stays below `rtol`, else `:violated`).

Half-vector-parameterised models are reciprocal by construction (expect ~0);
view-dependent diffuse/specular blends can break it if the blend is asymmetric.
"""
function reciprocity_error(model::BRDFModel; n_pairs::Int = 2000, rtol::Float64 = 1e-3)
    wis = _uniform_hemisphere(n_pairs)
    wos = _uniform_hemisphere(n_pairs)
    # Offset the second stream so wi != wo.
    max_abs = 0.0
    max_rel = 0.0
    for k in 1:n_pairs
        wi = wis[k]
        wo = wos[mod1(k + n_pairs ÷ 2, n_pairs)]
        a = evaluate(model, wi, wo)
        b = evaluate(model, wo, wi)
        for (x, y) in ((a.r, b.r), (a.g, b.g), (a.b, b.b))
            d = abs(Float64(x) - Float64(y))
            max_abs = max(max_abs, d)
            scale = max(abs(Float64(x)), abs(Float64(y)), 1e-8)
            max_rel = max(max_rel, d / scale)
        end
    end
    verdict = max_rel <= rtol ? :ok : :violated
    return (max_abs, verdict)
end

export brdf_rmse, brdf_rms_crwe, brdf_rmsle
export image_ssim, SSIM_LIGHT_DIRS
export reciprocity_error
