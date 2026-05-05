"""
Cook-Torrance microfacet BRDF with energy-conserving Fresnel-weighted diffuse.

    f(wi, wo) = (1 − F(wi,h)) · albedo/π  +  F(wi,h) · D(h) · G(wi,wo) / (4 · NdotL · NdotV)

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
- Cook & Torrance (1982)
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
    CookTorranceFresnelDiffuseModel <: BRDFModel

Cook-Torrance BRDF with Fresnel-weighted diffuse: `(1−F) · albedo/π + spec`.

The diffuse term is scaled by `(1 − F(VdotH))` per channel, which conserves
energy by preventing simultaneous full specular and full diffuse contributions.

# Fields
- `roughness`: Perceptual roughness ∈ [0, 1]. α = roughness² used internally.
- `F0`:        Fresnel reflectance at normal incidence (specular colour).
- `albedo`:    Diffuse reflectance colour. Set to `RGB(0,0,0)` for pure specular.
"""
struct CookTorranceFresnelDiffuseModel <: BRDFModel
    roughness::Float32
    F0::RGB{Float32}
    albedo::RGB{Float32}
end

"""Scalar convenience constructor (grey material)."""
CookTorranceFresnelDiffuseModel(roughness::Real, F0::Real, albedo::Real) =
    CookTorranceFresnelDiffuseModel(
        Float32(roughness),
        RGB{Float32}(F0, F0, F0),
        RGB{Float32}(albedo, albedo, albedo)
    )

# Microfacet helpers (_ggx_d, _smith_g1, _schlick) loaded from _microfacet_utils.jl

# ============================================================================
# Julia Evaluation
# ============================================================================

"""
    evaluate(model::CookTorranceFresnelDiffuseModel, wi, wo) -> RGB{Float32}

Evaluate the Cook-Torrance BRDF with (1−F)-weighted diffuse.
Surface normal is (0, 0, 1).
"""
function evaluate(model::CookTorranceFresnelDiffuseModel,
                  wi::SVector{3, Float32},
                  wo::SVector{3, Float32})::RGB{Float32}
    NdotL = wi[3]
    NdotV = wo[3]
    if NdotL <= 0.0f0 || NdotV <= 0.0f0
        return RGB{Float32}(0, 0, 0)
    end

    h     = normalize(wi + wo)
    NdotH = clamp(h[3], 0.0f0, 1.0f0)
    VdotH = clamp(dot(wo, h), 0.0f0, 1.0f0)

    alpha  = model.roughness * model.roughness
    alpha2 = max(alpha * alpha, 1.0f-6)

    D = _ggx_d(NdotH, alpha2)
    G = _smith_g1(NdotL, alpha2) * _smith_g1(NdotV, alpha2)

    Fr = _schlick(model.F0.r, VdotH)
    Fg = _schlick(model.F0.g, VdotH)
    Fb = _schlick(model.F0.b, VdotH)

    denom  = 4.0f0 * NdotL * NdotV
    spec_r = D * G * Fr / denom
    spec_g = D * G * Fg / denom
    spec_b = D * G * Fb / denom

    inv_pi = 1.0f0 / Float32(π)
    diff_r = (1.0f0 - Fr) * model.albedo.r * inv_pi
    diff_g = (1.0f0 - Fg) * model.albedo.g * inv_pi
    diff_b = (1.0f0 - Fb) * model.albedo.b * inv_pi

    return RGB{Float32}(
        max(0.0f0, diff_r + spec_r),
        max(0.0f0, diff_g + spec_g),
        max(0.0f0, diff_b + spec_b)
    )
end

# ============================================================================
# GLSL Shader
# ============================================================================

function glsl_shader(::CookTorranceFresnelDiffuseModel)::Tuple{String, Vector{String}}
    shader_path = joinpath(@__DIR__, "..", "rendering", "shaders", "cook_torrance_fresnel_diffuse.glsl")
    code = read(shader_path, String)
    return (code, ["u_roughness", "u_F0", "u_albedo"])
end

function shader_parameters(model::CookTorranceFresnelDiffuseModel)::Dict{String, Any}
    return Dict{String, Any}(
        "u_roughness" => model.roughness,
        "u_F0"        => Float32[model.F0.r, model.F0.g, model.F0.b],
        "u_albedo"    => Float32[model.albedo.r, model.albedo.g, model.albedo.b]
    )
end

# ============================================================================
# Registration
# ============================================================================

export CookTorranceFresnelDiffuseModel
register_model!(CookTorranceFresnelDiffuseModel)
