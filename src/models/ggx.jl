"""
GGX microfacet BRDF model with optional Lambertian diffuse component.

    f(wi, wo) = albedo/π  +  D(h) · G(wi,wo) · F(wi,h) / (4 · NdotL · NdotV)

Set albedo = BRDFRGB(0,0,0) for a pure-specular (metallic) surface.

Components:
- D: GGX/Trowbridge-Reitz normal distribution function
- G: Smith-GGX geometry term (separable form)
- F: Schlick Fresnel approximation

Roughness parameterisation: α = roughness²  (Disney/Unreal convention)

References:
- Walter et al., "Microfacet Models for Refraction" (2007)
- Schlick (1994)
- Burley (2012)
"""

using StaticArrays
using Colors
using LinearAlgebra

# ============================================================================
# Model Type
# ============================================================================

"""
    GGXModel <: BRDFModel

GGX microfacet BRDF with Lambertian diffuse component.

# Fields
- `roughness`: Perceptual roughness ∈ [0, 1]. α = roughness² used internally.
- `F0`:        Fresnel reflectance at normal incidence (specular colour).
- `albedo`:    Diffuse reflectance. Use BRDFRGB(0,0,0) for pure-specular (metallic) surfaces.
"""
struct GGXModel{T <: Real} <: BRDFModel
    roughness::T
    F0::BRDFRGB{T}
    albedo::BRDFRGB{T}
end

"""Scalar convenience constructor (grey specular, no diffuse)."""
function GGXModel(roughness::Real, F0::Real)
    T = float(promote_type(typeof(roughness), typeof(F0)))
    GGXModel(T(roughness), BRDFRGB{T}(F0, F0, F0), BRDFRGB{T}(zero(T), zero(T), zero(T)))
end

"""Scalar convenience constructor with grey diffuse."""
function GGXModel(roughness::Real, F0::Real, albedo::Real)
    T = float(promote_type(typeof(roughness), typeof(F0), typeof(albedo)))
    GGXModel(T(roughness), BRDFRGB{T}(F0, F0, F0), BRDFRGB{T}(albedo, albedo, albedo))
end

# Microfacet helpers (_ggx_d, _smith_g1, _schlick) loaded from _microfacet_utils.jl

# ============================================================================
# Julia Evaluation
# ============================================================================

"""
    evaluate(model::GGXModel, wi, wo)

Pure specular GGX BRDF. Surface normal is (0, 0, 1).
"""
function evaluate(model::GGXModel, wi, wo)
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
    r = model.albedo.r * inv_pi + spec_r
    g = model.albedo.g * inv_pi + spec_g
    b = model.albedo.b * inv_pi + spec_b
    return BRDFRGB(max(zero(r), r), max(zero(g), g), max(zero(b), b))
end

# ============================================================================
# Fit Spec
# ============================================================================

# 7 params: (roughness, F0 rgb, albedo rgb)
_fit_spec(::Type{GGXModel}) = _spec_microfacet_diffuse(GGXModel)

# ============================================================================
# Registration
# ============================================================================

register_model!(GGXModel)
