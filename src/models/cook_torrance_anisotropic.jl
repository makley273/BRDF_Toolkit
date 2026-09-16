"""
Anisotropic Cook-Torrance microfacet BRDF.

    f(wi, wo) = albedo/π  +  F(wi,h) · D(h) · G(wi,wo) / (4·NdotL·NdotV)

Components:
- D: Anisotropic Beckmann normal distribution (separate αx, αy)
- G: V-cavity geometric attenuation (independent of the slope distribution)
- F: Schlick Fresnel approximation

The two roughnesses αx, αy act along the surface tangent (x) and bitangent (y)
axes of the shading frame, allowing the highlight to be stretched into an
elongated lobe — useful for brushed metals, fabrics, hair.

Roughness parameterisation: αx = roughness_x², αy = roughness_y² (Disney remap).

References:
- Cook & Torrance (1982)
- Beckmann & Spizzichino (1963)
- Burley, "Physically-Based Shading at Disney" (2012) — anisotropic remap
"""

using StaticArrays
using Colors
using LinearAlgebra

# ============================================================================
# Model Type
# ============================================================================

"""
    CookTorranceAnisotropicModel <: BRDFModel

Anisotropic Cook-Torrance BRDF: anisotropic Beckmann NDF + V-cavity G +
Schlick Fresnel, with optional Lambertian diffuse.

# Fields
- `roughness_x`: Perceptual roughness along tangent ∈ [0, 1]. αx = roughness_x².
- `roughness_y`: Perceptual roughness along bitangent ∈ [0, 1]. αy = roughness_y².
- `F0`:          Fresnel reflectance at normal incidence.
- `albedo`:      Diffuse reflectance. Set to `BRDFRGB(0,0,0)` for pure specular.
"""
struct CookTorranceAnisotropicModel{T <: Real} <: BRDFModel
    roughness_x::T
    roughness_y::T
    F0::BRDFRGB{T}
    albedo::BRDFRGB{T}
end

"""Scalar convenience constructor (grey material)."""
function CookTorranceAnisotropicModel(rx::Real, ry::Real, F0::Real, albedo::Real)
    T = float(promote_type(typeof(rx), typeof(ry), typeof(F0), typeof(albedo)))
    CookTorranceAnisotropicModel(T(rx), T(ry), BRDFRGB{T}(F0, F0, F0), BRDFRGB{T}(albedo, albedo, albedo))
end

# ============================================================================
# Julia Evaluation
# ============================================================================

function evaluate(model::CookTorranceAnisotropicModel, wi, wo)
    NdotL = wi[3]
    NdotV = wo[3]
    if NdotL <= 0.0 || NdotV <= 0.0
        z = zero(model.roughness_x)
        return BRDFRGB(z, z, z)
    end

    h     = normalize(wi + wo)
    NdotH = clamp(h[3], 0.0, 1.0)
    VdotH = clamp(dot(wo, h), 0.0, 1.0)

    ax = max(model.roughness_x * model.roughness_x, 1e-4)
    ay = max(model.roughness_y * model.roughness_y, 1e-4)

    D = _beckmann_d_aniso(h[1], h[2], h[3], ax, ay)
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

# 8 params: (rough_x, rough_y, F0 rgb, albedo rgb)
_fit_spec(::Type{CookTorranceAnisotropicModel}) = _spec_microfacet_aniso_diffuse(CookTorranceAnisotropicModel)

# ============================================================================
# Registration
# ============================================================================

register_model!(CookTorranceAnisotropicModel)
