"""
Cook-Torrance microfacet BRDF with explicitly scaled diffuse component.

    f(wi, wo) = kd · albedo/π  +  F(wi,h) · D(h) · G(wi,wo) / (4 · NdotL · NdotV)

The scalar `kd ∈ [0, 1]` is a free fitting parameter that controls the
diffuse/specular balance independently of the albedo colour. Setting kd = 0
recovers pure-specular behaviour; kd = 1 reduces to the plain diffuse variant.

This gives the optimizer an explicit lever for the diffuse/specular split,
useful when fitting materials where the balance varies across samples.

Components:
- D: Beckmann normal distribution function
- G: V-cavity geometric attenuation
- F: Schlick Fresnel approximation

Roughness parameterisation: α = roughness²  (Disney/Unreal convention)

References:
- Cook & Torrance (1982)
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
    CookTorranceScaledDiffuseModel <: BRDFModel

Cook-Torrance BRDF with a free scalar diffuse weight: `kd · albedo/π + spec`.

# Fields
- `roughness`: Perceptual roughness ∈ [0, 1]. α = roughness² used internally.
- `F0`:        Fresnel reflectance at normal incidence (specular colour).
- `albedo`:    Diffuse reflectance colour.
- `kd`:        Diffuse weight scalar ∈ [0, 1]. Controls diffuse/specular balance.
"""
struct CookTorranceScaledDiffuseModel{T <: Real} <: BRDFModel
    roughness::T
    F0::BRDFRGB{T}
    albedo::BRDFRGB{T}
    kd::T
end

"""Scalar convenience constructor (grey material)."""
function CookTorranceScaledDiffuseModel(roughness::Real, F0::Real, albedo::Real, kd::Real)
    T = float(promote_type(typeof(roughness), typeof(F0), typeof(albedo), typeof(kd)))
    CookTorranceScaledDiffuseModel(T(roughness), BRDFRGB{T}(F0, F0, F0), BRDFRGB{T}(albedo, albedo, albedo), T(kd))
end

# Microfacet helpers (_ggx_d, _smith_g1, _schlick) loaded from _microfacet_utils.jl

# ============================================================================
# Julia Evaluation
# ============================================================================

"""
    evaluate(model::CookTorranceScaledDiffuseModel, wi, wo)

Evaluate the Cook-Torrance BRDF with kd-scaled diffuse. Surface normal is (0, 0, 1).
"""
function evaluate(model::CookTorranceScaledDiffuseModel, wi, wo)
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

    denom  = 4.0 * NdotL * NdotV
    spec_r = D * G * Fr / denom
    spec_g = D * G * Fg / denom
    spec_b = D * G * Fb / denom

    inv_pi = 1.0 / π
    diff_r = model.kd * model.albedo.r * inv_pi
    diff_g = model.kd * model.albedo.g * inv_pi
    diff_b = model.kd * model.albedo.b * inv_pi

    r = diff_r + spec_r
    g = diff_g + spec_g
    b = diff_b + spec_b
    return BRDFRGB(max(zero(r), r), max(zero(g), g), max(zero(b), b))
end

# ============================================================================
# Fit Spec
# ============================================================================

# 8 params: (roughness, F0 rgb, albedo rgb, kd)
_fit_spec(::Type{CookTorranceScaledDiffuseModel}) = _spec_microfacet_scaled(CookTorranceScaledDiffuseModel)

# ============================================================================
# Registration
# ============================================================================

register_model!(CookTorranceScaledDiffuseModel)
