"""
Error Maps — a first-class abstraction for visualizing fit error on the sphere.

An `ErrorMap` turns a *pair* of colors (measured reflectance `cm`, fitted-model
reflectance `cf`) at one direction into an error color. It is the visual analogue
of the scalar metrics in `analysis.jl` (`brdf_rmse`, `brdf_rms_crwe`): those
collapse the whole surface to one number, whereas an `ErrorMap` produces a
per-direction RGB that is sampled through the SAME universal LUT + shader as the
measured/fitted spheres (see `_ErrorWrapper` below).

`ErrorMap` is a sibling abstraction to `BRDFModel` / `BRDFData`: an abstract type
with a small required interface and a few concrete implementations. There is no
plugin registry/discovery (deliberately kept simple); the concrete maps the UI
offers live in `AVAILABLE_ERROR_MAPS`.

Interface for a concrete `ErrorMap`:
- `evaluate_error(em, cm, cf) -> RGB{Float32}`  (cm/cf expose `.r/.g/.b`)
- `error_map_label(em) -> String`               (shown in the dropdown / subtitle)
- `error_map_legend(em)` (optional)             (color-bar spec for the UI legend;
  defaults to `nothing` = no legend)

Location: src/errormaps/error_maps.jl
"""

using StaticArrays
using Colors

# ── Abstract type + interface ───────────────────────────────────────────────
abstract type ErrorMap end

"""
    evaluate_error(em::ErrorMap, cm, cf) -> RGB{Float32}

Map a measured color `cm` and a fitted-model color `cf` (both exposing `.r/.g/.b`)
at a single direction to an error color in `[0,1]³`. Pure function of `em`'s fields.
"""
function evaluate_error end

"""
    error_map_label(em::ErrorMap) -> String

Short human-readable name for the dropdown selector and sphere subtitle.
"""
function error_map_label end

"""
    error_map_legend(em::ErrorMap) -> Union{Nothing, NamedTuple}

Color-bar legend spec for the UI, or `nothing` (default) for no legend. The
NamedTuple has fields:
- `colors::Vector{String}` — CSS color stops, left to right (the UI joins them
  into a `linear-gradient`).
- `left`, `mid`, `right::String` — labels under the bar's ends and center.

Pure data (no DOM) so this file stays free of Bonito; the DOM lives in
`viz_blocks.jl` (`_legend_dom`).
"""
error_map_legend(::ErrorMap) = nothing

# RGB{Float32} → CSS "rgb(r,g,b)" for legend gradient stops.
_css_rgb(c::RGB{Float32}) =
    "rgb($(round(Int, 255 * clamp(c.r, 0, 1))),$(round(Int, 255 * clamp(c.g, 0, 1))),$(round(Int, 255 * clamp(c.b, 0, 1))))"

# Format 2^n as a compact factor string ("4" for 2.0, "2.8" for 1.5).
_factor_str(logdiff::Real) = let f = 2.0^logdiff
    isinteger(f) ? string(round(Int, f)) : string(round(f, sigdigits = 2))
end

# ── Shared helpers (self-contained; no dependency on analysis.jl) ────────────

# Rec.709 luminance.
@inline _err_lum(c) = 0.2126f0 * Float32(c.r) + 0.7152f0 * Float32(c.g) + 0.0722f0 * Float32(c.b)

# Classic "jet" colormap: t∈[0,1] → blue→cyan→green→yellow→red. Dependency-free.
@inline function _heat_color(t::Float32)
    t = clamp(t, 0.0f0, 1.0f0)
    r = clamp(1.5f0 - abs(4.0f0 * t - 3.0f0), 0.0f0, 1.0f0)
    g = clamp(1.5f0 - abs(4.0f0 * t - 2.0f0), 0.0f0, 1.0f0)
    b = clamp(1.5f0 - abs(4.0f0 * t - 1.0f0), 0.0f0, 1.0f0)
    return RGB{Float32}(r, g, b)
end

# Relative error with a denominator floor `eps` (avoids blow-up where data ≈ 0).
@inline _rel_err(a::Float32, b::Float32, eps::Float32) = abs(a - b) / (abs(a) + eps)

# ── Concrete error maps ─────────────────────────────────────────────────────
# Fields (shared by both): `max_rel` maps to the top of the scale (1.0 = 100%),
# `eps` is the relative-error denominator floor.

"""
    RelativeErrorRGB(; max_rel=1.0f0, eps=1.0f-3)

Per-channel relative error shown directly as RGB (each channel clamped to [0,1]).
"""
struct RelativeErrorRGB <: ErrorMap
    max_rel::Float32
    eps::Float32
end
RelativeErrorRGB(; max_rel::Real = 1.0f0, eps::Real = 1.0f-3) =
    RelativeErrorRGB(Float32(max_rel), Float32(eps))

function evaluate_error(em::RelativeErrorRGB, cm, cf)
    s = em.max_rel
    return RGB{Float32}(
        clamp(_rel_err(Float32(cm.r), Float32(cf.r), em.eps) / s, 0.0f0, 1.0f0),
        clamp(_rel_err(Float32(cm.g), Float32(cf.g), em.eps) / s, 0.0f0, 1.0f0),
        clamp(_rel_err(Float32(cm.b), Float32(cf.b), em.eps) / s, 0.0f0, 1.0f0),
    )
end

error_map_label(::RelativeErrorRGB) = "relative error (RGB)"

# Uniform error across all channels renders gray, so the legend is a black→white
# ramp; single-channel deviations tint toward that channel's color (not shown).
error_map_legend(em::RelativeErrorRGB) = (
    colors = [_css_rgb(RGB{Float32}(t, t, t)) for t in range(0.0f0, 1.0f0, length = 5)],
    left   = "0 %",
    mid    = "per-channel |rel. error|",
    right  = "≥ $(round(Int, 100 * em.max_rel)) %",
)

# Diverging blue-white-red colormap for SIGNED errors: t ∈ [-1, 1], white at 0,
# saturating to blue (negative) / red (positive). Piecewise-linear through white,
# which matters here: the generated LUT stores *colors*, so the GPU's trilinear
# filter interpolates colormap output — this map stays on (or near) the colormap
# under interpolation, unlike jet, whose blends produce off-scale muddy colors.
@inline function _diverging_color(t::Float32)
    t = clamp(t, -1.0f0, 1.0f0)
    if t >= 0.0f0
        return RGB{Float32}(1.0f0, 1.0f0 - t, 1.0f0 - t)   # white → red
    else
        return RGB{Float32}(1.0f0 + t, 1.0f0 + t, 1.0f0)   # white → blue
    end
end

"""
    SignedLogDifference(; max_logdiff=2.0f0, eps=1.0f-3)

Signed logarithmic luminance difference `log2(fitted+eps) - log2(measured+eps)`
on a diverging blue-white-red scale: **red = model too bright, blue = model too
dark, white = match**. `max_logdiff` sets full saturation (2.0 → a factor-4
over/underestimate saturates the color).

The log compression handles the BRDF's dynamic range: micro-errors in the
diffuse base stay visible next to macro-errors at the specular peak, and the
sign tells you which way to push the fit. Preferred map for tuning.
"""
struct SignedLogDifference <: ErrorMap
    max_logdiff::Float32
    eps::Float32
end
SignedLogDifference(; max_logdiff::Real = 2.0f0, eps::Real = 1.0f-3) =
    SignedLogDifference(Float32(max_logdiff), Float32(eps))

function evaluate_error(em::SignedLogDifference, cm, cf)
    s = log2(max(_err_lum(cf), 0.0f0) + em.eps) - log2(max(_err_lum(cm), 0.0f0) + em.eps)
    return _diverging_color(s / em.max_logdiff)
end

error_map_label(::SignedLogDifference) = "signed log difference (blue=dark, red=bright)"

error_map_legend(em::SignedLogDifference) = (
    colors = [_css_rgb(_diverging_color(t)) for t in range(-1.0f0, 1.0f0, length = 9)],
    left   = "model darker (÷$(_factor_str(em.max_logdiff)))",
    mid    = "match",
    right  = "model brighter (×$(_factor_str(em.max_logdiff)))",
)

"""
    SMAPEHeat(; eps=1.0f-3)

Symmetric Mean Absolute Percentage Error of the luminance,
`|fm - fa| / (fm + fa + eps)`, through the jet colormap. Strictly bounded in
[0, 1) by construction (no scale parameter to tune), treating over- and
underestimation symmetrically — but unsigned, so it can't tell you which.
"""
struct SMAPEHeat <: ErrorMap
    eps::Float32
end
SMAPEHeat(; eps::Real = 1.0f-3) = SMAPEHeat(Float32(eps))

function evaluate_error(em::SMAPEHeat, cm, cf)
    lm = max(_err_lum(cm), 0.0f0)
    lf = max(_err_lum(cf), 0.0f0)
    return _heat_color(abs(lm - lf) / (lm + lf + em.eps))
end

error_map_label(::SMAPEHeat) = "SMAPE (heat)"

error_map_legend(::SMAPEHeat) = (
    colors = [_css_rgb(_heat_color(t)) for t in range(0.0f0, 1.0f0, length = 9)],
    left   = "0 · match",
    mid    = "SMAPE",
    right  = "1 · max deviation",
)

# ── Bridge into the universal LUT pipeline ──────────────────────────────────
# Same trick as `_AnalyticalWrapper` in interactive_app.jl: an opaque `_data`
# whose `_brdf_value` returns the error color instead of a reflectance. Wrapped in
# a `MeasuredBRDF` it flows through the exact same
# `lookup` → `generate_lut` → `universal.glsl` path as the other two spheres
# — no changes to the LUT generator, shader, or renderer.
struct _ErrorWrapper
    measured::MeasuredBRDF
    model::BRDFModel
    errormap::ErrorMap
end

function _brdf_value(w::_ErrorWrapper,
                     ::Vector{Float32},
                     wi::SVector{3, Float32},
                     wo::SVector{3, Float32})
    cm = lookup(w.measured, wi, wo)        # measured reflectance
    cf = evaluate(w.model, wi, wo)         # fitted-model reflectance
    return evaluate_error(w.errormap, cm, cf)
end

n_samples_str(::_ErrorWrapper) = "error"

"""
    error_to_measured(measured, model, errormap, wavelengths, source_name) -> MeasuredBRDF

Wrap a measured BRDF + fitted model + `errormap` so the relative error renders
through the universal sphere pipeline (source format `:error`).
"""
function error_to_measured(measured::MeasuredBRDF, model::BRDFModel, errormap::ErrorMap,
                           wavelengths::Vector{Float32}, source_name::String)::MeasuredBRDF
    return MeasuredBRDF(
        _ErrorWrapper(measured, model, errormap),
        wavelengths,
        "err:" * source_name;
        source_format = :error,
        status = :ok,
    )
end

# ── Available error maps for the UI dropdown (no registry, by design) ────────
# SignedLogDifference first → default selection: signed + log-compressed is the
# most actionable map for tuning a fit (see its docstring).
const AVAILABLE_ERROR_MAPS = ErrorMap[
    SignedLogDifference(),
    RelativeErrorRGB(),
    SMAPEHeat(),
]
