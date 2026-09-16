"""
GGX microfacet BRDF with energy-conserving Fresnel-weighted diffuse component.

    f(wi, wo) = (1 − F(wi,h)) · albedo/π  +  D(h) · G(wi,wo) · F(wi,h) / (4 · NdotL · NdotV)

The factor (1 − F) ensures energy conservation: at grazing angles where the
specular Fresnel term approaches 1, the diffuse contribution fades to zero.
This mirrors the physical intuition that light either reflects specularly or
transmits into the surface for subsurface scattering — it cannot do both.

Per-channel (RGB) Fresnel is used, so albedo tinting is modulated independently
per wavelength.

Components:
- D: GGX/Trowbridge-Reitz normal distribution function
- G: Smith-GGX geometry term (separable form)
- F: Schlick Fresnel approximation

Roughness parameterisation: α = roughness²  (Disney/Unreal convention)

References:
- Walter et al., "Microfacet Models for Refraction" (2007)
- Schlick (1994)
- Burley (2012) — Disney principled BRDF (uses similar (1−F) weighting)
"""

using StaticArrays
using Colors
using LinearAlgebra

# ============================================================================
# Model Type
# ============================================================================

"""
    GGXFresnelDiffuseModel <: BRDFModel

GGX microfacet BRDF with Fresnel-weighted diffuse: `(1−F) · albedo/π + spec`.

The diffuse term is scaled by `(1 − F(VdotH))` per channel, which conserves
energy by preventing simultaneous full specular and full diffuse contributions.

# Fields
- `roughness`: Perceptual roughness ∈ [0, 1]. α = roughness² used internally.
- `F0`:        Fresnel reflectance at normal incidence (specular colour).
- `albedo`:    Diffuse reflectance colour. Set to `BRDFRGB(0,0,0)` for pure specular.
"""
struct GGXFresnelDiffuseModel{T <: Real} <: BRDFModel
    roughness::T
    F0::BRDFRGB{T}
    albedo::BRDFRGB{T}
end

"""Scalar convenience constructor (grey material)."""
function GGXFresnelDiffuseModel(roughness::Real, F0::Real, albedo::Real)
    T = float(promote_type(typeof(roughness), typeof(F0), typeof(albedo)))
    GGXFresnelDiffuseModel(T(roughness), BRDFRGB{T}(F0, F0, F0), BRDFRGB{T}(albedo, albedo, albedo))
end

# Microfacet helpers (_ggx_d, _smith_g1, _schlick) loaded from _microfacet_utils.jl

# ============================================================================
# Julia Evaluation
# ============================================================================

"""
    evaluate(model::GGXFresnelDiffuseModel, wi, wo)

Evaluate the GGX BRDF with (1−F)-weighted diffuse. Surface normal is (0, 0, 1).
"""
function evaluate(model::GGXFresnelDiffuseModel, wi, wo)
    NdotL = wi[3]
    NdotV = wo[3]
    if NdotL <= 0.0 || NdotV <= 0.0
        z = zero(model.roughness)
        return BRDFRGB(z, z, z)
    end

    h     = normalize(wi + wo)
    NdotH = clamp(h[3], 0.0, 1.0)
    VdotH = clamp(dot(wo, h), 0.0, 1.0)

    alpha  = model.roughness * model.roughness
    alpha2 = max(alpha * alpha, 1e-6)

    D = _ggx_d(NdotH, alpha2)
    G = _smith_g1(NdotL, alpha2) * _smith_g1(NdotV, alpha2)

    Fr = _schlick(model.F0.r, VdotH)
    Fg = _schlick(model.F0.g, VdotH)
    Fb = _schlick(model.F0.b, VdotH)

    denom  = 4.0 * NdotL * NdotV
    spec_r = D * G * Fr / denom
    spec_g = D * G * Fg / denom
    spec_b = D * G * Fb / denom

    inv_pi = 1.0 / π
    diff_r = (1.0 - Fr) * model.albedo.r * inv_pi
    diff_g = (1.0 - Fg) * model.albedo.g * inv_pi
    diff_b = (1.0 - Fb) * model.albedo.b * inv_pi

    r = diff_r + spec_r
    g = diff_g + spec_g
    b = diff_b + spec_b
    return BRDFRGB(max(zero(r), r), max(zero(g), g), max(zero(b), b))
end

# ============================================================================
# Fit Spec
# ============================================================================

# 7 params: (roughness, F0 rgb, albedo rgb)
_fit_spec(::Type{GGXFresnelDiffuseModel}) = _spec_microfacet_diffuse(GGXFresnelDiffuseModel)

# ============================================================================
# Registration
# ============================================================================

register_model!(GGXFresnelDiffuseModel)
