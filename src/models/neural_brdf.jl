"""
Neural BRDF model — a small multi-layer perceptron (MLP) fit to one measured
material.

This is the *purely-neural* single-MLP approach: the network maps a pair of
directions (via the Rusinkiewicz parameterization) directly to an RGB reflectance
value, with no underlying analytical model. It follows Sztrajman et al. 2021
("Neural BRDF Representation and Importance Sampling") for the architecture
(shallow FC net, non-negativity output activation), combined with the
reciprocity-by-construction input mapping of Hofherr et al. 2025
("On Neural BRDFs", Eq. 6).

# Input mapping (reciprocity by construction)
The directions are mapped to reciprocity-invariant features by `_brdf_features`,
in one of two modes (`anisotropic`):

    isotropic   (4D): [θ_h, θ_d, sin θ_h·cos 2φ_d, sin θ_h·sin 2φ_d]
    anisotropic (6D): [h_x, h_y, h_z, θ_d, sin θ_h·cos 2φ_d, sin θ_h·sin 2φ_d]

Swapping `wi ↔ wo` leaves the half-vector `H` (hence `θ_h` and `(h_x,h_y,h_z)`)
and `θ_d` unchanged, and sends `φ_d → φ_d + π`; the second-harmonic features
`(cos 2φ_d, sin 2φ_d)` are invariant under that shift. They enter the input scaled by
`sin θ_h`, which fades the azimuth out as `H → N`, where `φ_d` stops being meaningful.
So every input feature is
invariant, and Helmholtz reciprocity `f(wi, wo) = f(wo, wi)` holds *exactly* for
any weights, in both modes. The isotropic mode drops `φ_h`; the anisotropic mode
keeps it via the half-vector components, so it can represent brushed/anisotropic
materials (at the cost of a wider input and more training data).

# Weights, not Lux
The model stores plain `Float32` weight matrices and biases (extracted from the
trained Lux network by `NeuralBRDFTrainer`). `evaluate` is a manual forward pass,
so it is fast and allocation-light even when the universal LUT generator calls it
~1M times per render — and it carries no heavyweight ML object.

The non-negativity of the output is guaranteed by `out_act` (softplus by default).
There is no per-model GLSL: the universal 4D-LUT renderer samples any model through
`evaluate`.
"""

using StaticArrays
using Colors
using LinearAlgebra

# ============================================================================
# Model Type
# ============================================================================

"""
    NeuralBRDFModel <: BRDFModel

Trained MLP BRDF for a single material.

# Fields
- `Ws`:          Per-layer weight matrices, each `(n_out × n_in)`.
- `bs`:          Per-layer bias vectors, each length `n_out`.
- `hidden_act`:  Hidden-layer activation (`:relu`, `:gelu`, `:tanh`, `:softplus`).
- `out_act`:     Output activation enforcing non-negativity
                 (`:softplus`, `:exp`, `:square`, `:abs`, `:relu`).
- `arch`:        Full layer-width vector, e.g. `[4, 21, 21, 3]` (provenance/UI).
- `anisotropic`: Selects the input feature mapping (`_brdf_features`):
                 `false` → isotropic 4-vector `[θ_h, θ_d, sin θ_h·cos 2φ_d, sin θ_h·sin 2φ_d]`;
                 `true`  → anisotropic 6-vector `[h_x, h_y, h_z, θ_d, sin θ_h·cos 2φ_d, sin θ_h·sin 2φ_d]`
                 which keeps the half-vector azimuth `φ_h` so anisotropy can be
                 represented. Both mappings are exactly reciprocity-invariant.
                 Must agree with `arch[1]` (4 or 6).
"""
struct NeuralBRDFModel <: BRDFModel
    Ws::Vector{Matrix{Float32}}
    bs::Vector{Vector{Float32}}
    hidden_act::Symbol
    out_act::Symbol
    arch::Vector{Int}
    anisotropic::Bool
end

# Backward-compatible constructor: omitting `anisotropic` defaults to isotropic.
NeuralBRDFModel(Ws, bs, hidden_act, out_act, arch) =
    NeuralBRDFModel(Ws, bs, hidden_act, out_act, arch, false)

# ============================================================================
# Activations (scalar; broadcast at call sites)
# ============================================================================

# Named scalar activations. These are plain (singleton, isbits) functions so they
# can be handed straight to Lux `Dense` layers and compiled into CUDA kernels —
# unlike a `sym`-capturing closure, whose `Symbol` field is not isbits and breaks
# GPU compilation. `_apply_hidden`/`_apply_out` dispatch on the symbol to exactly
# these functions, so the manual forward pass and the Lux network stay identical.
@inline _relu(x) = max(x, zero(x))
@inline _square(x) = x * x
# Numerically stable softplus: log(1 + e^x) without overflow for large x.
@inline _softplus(x) = max(x, zero(x)) + log1p(exp(-abs(x)))
# GELU (tanh approximation, matches common MLP usage).
@inline function _gelu(x)
    c = oftype(x, 0.7978845608028654)   # sqrt(2/π)
    return oftype(x, 0.5) * x * (one(x) + tanh(c * (x + oftype(x, 0.044715) * x^3)))
end

"""
    _hidden_activation(sym) -> Function

Map a hidden-activation symbol to its concrete scalar function (a singleton, so
the result is isbits and GPU-safe).
"""
@inline function _hidden_activation(sym::Symbol)
    sym === :relu     && return _relu
    sym === :gelu     && return _gelu
    sym === :tanh     && return tanh
    sym === :softplus && return _softplus
    throw(ArgumentError("unknown hidden activation: $sym"))
end

"""
    _output_activation(sym) -> Function

Map a non-negativity output-activation symbol to its concrete scalar function.
"""
@inline function _output_activation(sym::Symbol)
    sym === :softplus && return _softplus
    sym === :exp      && return exp
    sym === :square   && return _square
    sym === :abs      && return abs
    sym === :relu     && return _relu
    throw(ArgumentError("unknown output activation: $sym"))
end

@inline _apply_hidden(sym::Symbol, x) = _hidden_activation(sym)(x)
@inline _apply_out(sym::Symbol, x)    = _output_activation(sym)(x)

# Activation choices exposed to the trainer / UI.
const NEURAL_HIDDEN_ACTS = (:relu, :gelu, :tanh, :softplus)
const NEURAL_OUTPUT_ACTS = (:softplus, :exp, :square, :abs, :relu)

# ============================================================================
# Reciprocity-exact input features
# ============================================================================

"""
    _brdf_features(anisotropic, wi, wo) -> SVector{4|6, Float32}

Map directions to the network input features. Every component is invariant under
`wi ↔ wo` *bit-for-bit* in Float32, so the fitted BRDF satisfies Helmholtz
reciprocity exactly (not just approximately) regardless of the trained weights.
Each feature is built only from swap-symmetric quantities:

- the sum `s = wi + wo` is commutative, so `H = s/‖s‖` (hence `cos θ_h` and the
  local components `(h_x, h_y, h_z)`) is bit-identical under the swap;
- `cos θ_d = ‖s‖ / 2` (since `‖wi + wo‖ = 2 cos θ_d` for unit directions) — built
  from the same commutative sum, so also bit-identical;
- the difference `D = wi − wo` satisfies `wo − wi == −D` exactly in IEEE
  arithmetic, and the second-harmonic features `(cos 2φ_d, sin 2φ_d) =
  ((Dx²−Dy²)/r², 2 Dx Dy / r²)` are even in `D`, hence bit-identical under the
  swap (and circularly continuous, unlike `φ_d` itself). `D` is `2×` the component
  of `wi` perpendicular to `H`, so it carries the Rusinkiewicz difference azimuth
  `φ_d` unchanged.

Modes (selected by `anisotropic`):
- `false` → `[θ_h, θ_d, sin θ_h·cos 2φ_d, sin θ_h·sin 2φ_d]` (isotropic; `φ_h` dropped).
- `true`  → `[h_x, h_y, h_z, θ_d, sin θ_h·cos 2φ_d, sin θ_h·sin 2φ_d]`; the half-vector carries
  both `θ_h` and the azimuth `φ_h`, so anisotropy can be represented.

The `(t, b, h)` frame matches `_sample_directions_halfangle` / the universal-LUT
generator, so the `φ_d`/`φ_h` conventions are consistent across sampling, LUT
generation, and evaluation.
"""
@inline function _brdf_features(anisotropic::Bool,
                                wi::SVector{3,Float32}, wo::SVector{3,Float32})
    N = SVector{3,Float32}(0, 0, 1)

    h = wi + wo
    hl = norm(h)
    h = hl > 1.0f-8 ? h / hl : N

    # Frame (t, b, h) built exactly as in the half-angle sampler.
    Nh = dot(N, h)
    t_raw = N - Nh * h
    tl = norm(t_raw)
    local t, b
    if tl > 1.0f-6
        t = t_raw / tl
        b = cross(h, t)
    else
        t = SVector{3,Float32}(1, 0, 0)
        b = SVector{3,Float32}(0, 1, 0)
    end

    # cos θ_d from the commutative sum: ‖wi + wo‖ = 2 cos θ_d for unit vectors.
    # NOT dot(wi, h) — that is only *mathematically* swap-symmetric; built from
    # `hl` it is bit-identical under wi ↔ wo, which exact reciprocity requires.
    cos_td = clamp(hl * 0.5f0, 0.0f0, 1.0f0)
    td = acos(cos_td)
    cos_th = clamp(h[3], -1.0f0, 1.0f0)
    th = acos(cos_th)

    # Second-harmonic azimuth features from the DIFFERENCE vector D = wi − wo
    # (2× the component of wi perpendicular to H, so it carries φ_d; the 2×
    # scale cancels in the ratios). Under the swap D → −D *exactly* in IEEE
    # arithmetic, so (dx, dy) → (−dx, −dy) bit-for-bit and the even features
    # below are bit-identical — using dot(wi, t) directly instead breaks
    # bit-exactness (swap-symmetric only up to rounding).
    # With r = ‖D_⊥‖: cos 2φ_d = (dx² − dy²)/r², sin 2φ_d = 2 dx dy / r².
    # No trig, and they fold φ_d → φ_d + π automatically. At θ_d ≈ 0 (φ_d
    # undefined) fall back to φ_d = 0 — a measure-zero direction.
    D = wi - wo
    dx = dot(D, t); dy = dot(D, b)
    r2 = dx * dx + dy * dy
    cos_2pd, sin_2pd = if r2 > 1.0f-12
        inv = 1.0f0 / r2
        ((dx * dx - dy * dy) * inv, (2.0f0 * dx * dy) * inv)
    else
        (1.0f0, 0.0f0)
    end

    if anisotropic
        return SVector{6,Float32}(h[1], h[2], h[3], td, sin(th)*cos_2pd, sin(th)*sin_2pd)
    else
        return SVector{4,Float32}(th, td, sin(th)*cos_2pd, sin(th)*sin_2pd)
    end
end

# Input dimensionality for a given mode (4 isotropic, 6 anisotropic).
@inline _feature_dim(anisotropic::Bool) = anisotropic ? 6 : 4

# ============================================================================
# Forward pass / evaluation
# ============================================================================

"""
    _forward(model, theta) -> SVector{3}

Manual MLP forward pass over the stored `Float32` weights. Generic in the input
element type so plain `Float32` (rendering) and `ForwardDiff.Dual` both work.
"""
function _forward(model::NeuralBRDFModel, theta::AbstractVector)
    x = collect(theta)
    n = length(model.Ws)
    @inbounds for i in 1:n
        x = model.Ws[i] * x .+ model.bs[i]
        if i < n
            x = _apply_hidden.(model.hidden_act, x)
        else
            x = _apply_out.(model.out_act, x)
        end
    end
    return x
end

"""
    evaluate(model::NeuralBRDFModel, wi, wo) -> BRDFRGB

Evaluate the trained MLP. Returns zero below the horizon; otherwise a
non-negative RGB reflectance (non-negativity guaranteed by `out_act`).
"""
function evaluate(model::NeuralBRDFModel, wi, wo)
    if wi[3] <= 0.0 || wo[3] <= 0.0
        return BRDFRGB(0.0f0, 0.0f0, 0.0f0)
    end
    wi_f = SVector{3,Float32}(wi[1], wi[2], wi[3])
    wo_f = SVector{3,Float32}(wo[1], wo[2], wo[3])
    feats = _brdf_features(model.anisotropic, wi_f, wo_f)
    y = _forward(model, feats)
    return BRDFRGB(y[1], y[2], y[3])
end

# ============================================================================
# Parameter readout (no GLSL — universal LUT renders this model)
# ============================================================================

"""
    display_parameters(model::NeuralBRDFModel) -> Dict{String, Any}

Custom UI readout: the generic fallback would dump the raw weight matrices, so
this overrides it with a human-readable architecture summary instead.
"""
function display_parameters(model::NeuralBRDFModel)::Dict{String, Any}
    n_params = sum(length, model.Ws) + sum(length, model.bs)
    return Dict{String, Any}(
        "architecture"   => join(string.(model.arch), "→"),
        "mode"           => model.anisotropic ? "anisotropic" : "isotropic",
        "hidden_act"     => String(model.hidden_act),
        "output_act"     => String(model.out_act),
        "num_parameters" => n_params,
    )
end

# ============================================================================
# Registration
# ============================================================================

register_model!(NeuralBRDFModel)
