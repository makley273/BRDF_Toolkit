"""
Cook-Torrance microfacet BRDF model.

Implements a diffuse + specular BRDF:
    f(wi, wo) = albedo/π  +  F(wi,h) · D(h) · G(wi,wo) / (4·NdotL·NdotV)

Components:
- D: GGX/Trowbridge-Reitz normal distribution function
- G: Smith-GGX height-correlated geometry term (separable form)
- F: Schlick Fresnel approximation

Roughness parameterisation follows the Disney/Unreal convention:
    α = roughness²  (perceptual roughness → linear roughness)

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
    CookTorranceModel <: BRDFModel

Microfacet BRDF with GGX NDF, Smith-GGX geometry, and Schlick Fresnel.

# Fields
- `roughness`: Perceptual roughness ∈ [0, 1]. α = roughness² is used internally.
- `F0`:        Fresnel reflectance at normal incidence (specular colour).
- `albedo`:    Diffuse reflectance. Set to `RGB(0,0,0)` for pure specular.
"""
struct CookTorranceModel <: BRDFModel
    roughness::Float32
    F0::RGB{Float32}
    albedo::RGB{Float32}
end

"""Scalar convenience constructor (grey material)."""
function CookTorranceModel(roughness::Real, F0::Real, albedo::Real)
    CookTorranceModel(
        Float32(roughness),
        RGB{Float32}(F0, F0, F0),
        RGB{Float32}(albedo, albedo, albedo)
    )
end

# ============================================================================
# Internal microfacet helpers
# ============================================================================

"""GGX normal distribution function. `alpha2` = α⁴ (roughness⁴)."""
@inline function _ggx_d(NdotH::Float32, alpha2::Float32)::Float32
    d = NdotH * NdotH * (alpha2 - 1.0f0) + 1.0f0
    return alpha2 / (Float32(π) * d * d)
end

"""Smith-GGX masking function (single direction). `alpha2` = α⁴."""
@inline function _smith_g1(NdotV::Float32, alpha2::Float32)::Float32
    return 2.0f0 * NdotV / (NdotV + sqrt(alpha2 + (1.0f0 - alpha2) * NdotV * NdotV))
end

"""Schlick Fresnel approximation (scalar F0)."""
@inline function _schlick(F0::Float32, VdotH::Float32)::Float32
    x = 1.0f0 - VdotH
    return F0 + (1.0f0 - F0) * x * x * x * x * x
end

# ============================================================================
# Julia Evaluation
# ============================================================================

"""
    evaluate(model::CookTorranceModel, wi, wo) -> RGB{Float32}

Evaluate the Cook-Torrance BRDF at incident direction `wi` and outgoing `wo`.
Surface normal is (0, 0, 1).
"""
function evaluate(model::CookTorranceModel,
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

    # Disney roughness remap: α = roughness², α² = roughness⁴
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

    diff = Float32(1.0 / π)
    r = model.albedo.r * diff + spec_r
    g = model.albedo.g * diff + spec_g
    b = model.albedo.b * diff + spec_b

    return RGB{Float32}(max(0.0f0, r), max(0.0f0, g), max(0.0f0, b))
end

# ============================================================================
# GLSL Shader
# ============================================================================

function glsl_shader(::CookTorranceModel)::Tuple{String, Vector{String}}
    shader_path = joinpath(@__DIR__, "..", "rendering", "shaders", "cook_torrance.glsl")
    code = read(shader_path, String)
    uniforms = ["u_roughness", "u_F0", "u_albedo"]
    return (code, uniforms)
end

function shader_parameters(model::CookTorranceModel)::Dict{String, Any}
    return Dict{String, Any}(
        "u_roughness" => model.roughness,
        "u_F0"        => Float32[model.F0.r, model.F0.g, model.F0.b],
        "u_albedo"    => Float32[model.albedo.r, model.albedo.g, model.albedo.b]
    )
end

# ============================================================================
# Registration
# ============================================================================

export CookTorranceModel
register_model!(CookTorranceModel)
