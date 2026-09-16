"""
GGX microfacet BRDF — pure specular, no diffuse component.

    f(wi, wo) = D(h) · G(wi,wo) · F(wi,h) / (4 · NdotL · NdotV)

Use this variant when fitting metallic or purely reflective materials,
or as a baseline to isolate the specular lobe.

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
    GGXSpecularModel <: BRDFModel

Pure-specular GGX microfacet BRDF. No diffuse component.

# Fields
- `roughness`: Perceptual roughness ∈ [0, 1]. α = roughness² used internally.
- `F0`:        Fresnel reflectance at normal incidence (specular colour).
"""
struct GGXSpecularModel{T <: Real} <: BRDFModel
    roughness::T
    F0::BRDFRGB{T}
end

"""Scalar convenience constructor (grey specular)."""
function GGXSpecularModel(roughness::Real, F0::Real)
    T = float(promote_type(typeof(roughness), typeof(F0)))
    GGXSpecularModel(T(roughness), BRDFRGB{T}(F0, F0, F0))
end

# Microfacet helpers (_ggx_d, _smith_g1, _schlick) loaded from _microfacet_utils.jl

# ============================================================================
# Julia Evaluation
# ============================================================================

"""
    evaluate(model::GGXSpecularModel, wi, wo)

Evaluate the pure-specular GGX BRDF. Surface normal is (0, 0, 1).
"""
function evaluate(model::GGXSpecularModel, wi, wo)
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

    denom = 4.0 * NdotL * NdotV
    r = D * G * Fr / denom
    g = D * G * Fg / denom
    b = D * G * Fb / denom
    return BRDFRGB(max(zero(r), r), max(zero(g), g), max(zero(b), b))
end

# ============================================================================
# Fit Spec
# ============================================================================

# 4 params: (roughness, F0 rgb)
_fit_spec(::Type{GGXSpecularModel}) = _spec_microfacet_specular(GGXSpecularModel)

# ============================================================================
# Registration
# ============================================================================

register_model!(GGXSpecularModel)
