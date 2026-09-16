"""
Cook-Torrance microfacet BRDF — pure specular, no diffuse component.

    f(wi, wo) = F(wi,h) · D(h) · G(wi,wo) / (4 · NdotL · NdotV)

Use this variant when fitting metallic or purely reflective materials,
or as a baseline to isolate the specular lobe.

Components:
- D: Beckmann normal distribution function
- G: V-cavity geometric attenuation
- F: Schlick Fresnel approximation

Roughness parameterisation: α = roughness²  (Disney/Unreal convention)

References:
- Cook & Torrance (1982)
- Walter et al., "Microfacet Models for Refraction" (2007) — GGX NDF & Smith G
- Schlick (1994) — Fresnel approximation
- Burley (2012) — Disney roughness remap
"""

using StaticArrays
using Colors
using LinearAlgebra

# ============================================================================
# Model Type
# ============================================================================

"""
    CookTorranceSpecularModel <: BRDFModel

Pure-specular Cook-Torrance microfacet BRDF. No diffuse component.

# Fields
- `roughness`: Perceptual roughness ∈ [0, 1]. α = roughness² used internally.
- `F0`:        Fresnel reflectance at normal incidence (specular colour).
"""
struct CookTorranceSpecularModel{T <: Real} <: BRDFModel
    roughness::T
    F0::BRDFRGB{T}
end

"""Scalar convenience constructor (grey specular)."""
function CookTorranceSpecularModel(roughness::Real, F0::Real)
    T = float(promote_type(typeof(roughness), typeof(F0)))
    CookTorranceSpecularModel(T(roughness), BRDFRGB{T}(F0, F0, F0))
end

# Microfacet helpers (_ggx_d, _smith_g1, _schlick) loaded from _microfacet_utils.jl

# ============================================================================
# Julia Evaluation
# ============================================================================

"""
    evaluate(model::CookTorranceSpecularModel, wi, wo)

Evaluate the pure-specular Cook-Torrance BRDF. Surface normal is (0, 0, 1).
"""
function evaluate(model::CookTorranceSpecularModel, wi, wo)
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

    D = _beckmann_d(NdotH, alpha2)
    G = _vcavity_g(NdotL, NdotV, NdotH, VdotH)

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
_fit_spec(::Type{CookTorranceSpecularModel}) = _spec_microfacet_specular(CookTorranceSpecularModel)

# ============================================================================
# Registration
# ============================================================================

register_model!(CookTorranceSpecularModel)
