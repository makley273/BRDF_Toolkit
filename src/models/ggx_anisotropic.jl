"""
Anisotropic GGX microfacet BRDF.

    f(wi, wo) = albedo/π  +  D(h) · G(wi,wo) · F(wi,h) / (4·NdotL·NdotV)

Components:
- D: Anisotropic GGX/Trowbridge-Reitz NDF (separate αx, αy)
- G: Anisotropic Smith-GGX masking-shadowing
- F: Schlick Fresnel approximation

The two roughnesses αx, αy act along the tangent (x) and bitangent (y) axes,
producing the characteristic stretched highlights of brushed metals and hair.

Roughness parameterisation: αx = roughness_x², αy = roughness_y² (Disney remap).

References:
- Walter et al., "Microfacet Models for Refraction" (2007)
- Burley, "Physically-Based Shading at Disney" (2012) — anisotropic remap
- Heitz, "Understanding the Masking-Shadowing Function..." (2014)
"""

using StaticArrays
using Colors
using LinearAlgebra

# ============================================================================
# Model Type
# ============================================================================

"""
    GGXAnisotropicModel <: BRDFModel

Anisotropic GGX BRDF: anisotropic GGX NDF + anisotropic Smith masking +
Schlick Fresnel, with optional Lambertian diffuse.

# Fields
- `roughness_x`: Perceptual roughness along tangent ∈ [0, 1]. αx = roughness_x².
- `roughness_y`: Perceptual roughness along bitangent ∈ [0, 1]. αy = roughness_y².
- `F0`:          Fresnel reflectance at normal incidence.
- `albedo`:      Diffuse reflectance. Set to `BRDFRGB(0,0,0)` for pure specular.
"""
struct GGXAnisotropicModel{T <: Real} <: BRDFModel
    roughness_x::T
    roughness_y::T
    F0::BRDFRGB{T}
    albedo::BRDFRGB{T}
end

"""Scalar convenience constructor (grey material)."""
function GGXAnisotropicModel(rx::Real, ry::Real, F0::Real, albedo::Real)
    T = float(promote_type(typeof(rx), typeof(ry), typeof(F0), typeof(albedo)))
    GGXAnisotropicModel(T(rx), T(ry), BRDFRGB{T}(F0, F0, F0), BRDFRGB{T}(albedo, albedo, albedo))
end

# ============================================================================
# Julia Evaluation
# ============================================================================

function evaluate(model::GGXAnisotropicModel, wi, wo)
    NdotL = wi[3]
    NdotV = wo[3]
    if NdotL <= 0.0 || NdotV <= 0.0
        z = zero(model.roughness_x)
        return BRDFRGB(z, z, z)
    end

    h     = normalize(wi + wo)
    VdotH = clamp(dot(wo, h), 0.0, 1.0)

    ax = max(model.roughness_x * model.roughness_x, 1e-4)
    ay = max(model.roughness_y * model.roughness_y, 1e-4)

    D = _ggx_d_aniso(h[1], h[2], h[3], ax, ay)
    G = _smith_g1_aniso(wi[1], wi[2], wi[3], ax, ay) *
        _smith_g1_aniso(wo[1], wo[2], wo[3], ax, ay)

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

# 8 params: (rough_x, rough_y, F0 rgb, albedo rgb)
_fit_spec(::Type{GGXAnisotropicModel}) = _spec_microfacet_aniso_diffuse(GGXAnisotropicModel)

# ============================================================================
# Registration
# ============================================================================

register_model!(GGXAnisotropicModel)
