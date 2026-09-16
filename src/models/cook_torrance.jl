"""
Cook-Torrance microfacet BRDF model.

Implements a diffuse + specular BRDF:
    f(wi, wo) = albedo/π  +  F(wi,h) · D(h) · G(wi,wo) / (4·NdotL·NdotV)

Components (classic Cook-Torrance 1982):
- D: Beckmann normal distribution function
- G: V-cavity geometric attenuation `min(1, 2·NH·NV/VH, 2·NH·NL/VH)`
- F: Schlick Fresnel approximation

Roughness parameterisation follows the Disney/Unreal convention:
    α = roughness²  (perceptual roughness → linear roughness)

References:
- Cook & Torrance, "A Reflectance Model for Computer Graphics" (1982)
- Beckmann & Spizzichino, "The scattering of electromagnetic waves..." (1963)
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
    CookTorranceModel <: BRDFModel

Classic Cook-Torrance microfacet BRDF: Beckmann NDF + V-cavity geometry +
Schlick Fresnel. Distinct from `GGXModel`, which uses the GGX NDF with Smith
masking (longer highlight tails).

# Fields
- `roughness`: Perceptual roughness ∈ [0, 1]. α = roughness² is used internally.
- `F0`:        Fresnel reflectance at normal incidence (specular colour).
- `albedo`:    Diffuse reflectance. Set to `BRDFRGB(0,0,0)` for pure specular.
"""
struct CookTorranceModel{T <: Real} <: BRDFModel
    roughness::T
    F0::BRDFRGB{T}
    albedo::BRDFRGB{T}
end

"""Scalar convenience constructor (grey material)."""
function CookTorranceModel(roughness::Real, F0::Real, albedo::Real)
    T = promote_type(typeof(roughness), typeof(F0), typeof(albedo))
    T_target = T <: Integer ? Float64 : T 
    CookTorranceModel(
        T_target(roughness),
        BRDFRGB{T_target}(F0, F0, F0),
        BRDFRGB{T_target}(albedo, albedo, albedo)
    )
end

# Microfacet helpers (_ggx_d, _smith_g1, _schlick) loaded from _microfacet_utils.jl

# ============================================================================
# Julia Evaluation
# ============================================================================

"""
    evaluate(model::CookTorranceModel, wi, wo)

Evaluate the Cook-Torrance BRDF at incident direction `wi` and outgoing `wo`.
Surface normal is (0, 0, 1).
"""
function evaluate(model::CookTorranceModel, wi, wo)
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

    diff = 1.0 / π
    r = model.albedo.r * diff + spec_r
    g = model.albedo.g * diff + spec_g
    b = model.albedo.b * diff + spec_b

    return BRDFRGB(max(zero(r), r), max(zero(g), g), max(zero(b), b))
end

# ============================================================================
# Fit Spec
# ============================================================================

# 7 params: (roughness, F0 rgb, albedo rgb)
_fit_spec(::Type{CookTorranceModel}) = _spec_microfacet_diffuse(CookTorranceModel)

# ============================================================================
# Registration
# ============================================================================

register_model!(CookTorranceModel)
