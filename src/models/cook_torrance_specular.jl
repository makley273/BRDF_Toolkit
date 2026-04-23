"""
Cook-Torrance microfacet BRDF — pure specular, no diffuse component.

    f(wi, wo) = F(wi,h) · D(h) · G(wi,wo) / (4 · NdotL · NdotV)

Use this variant when fitting metallic or purely reflective materials,
or as a baseline to isolate the specular lobe.

Components:
- D: GGX/Trowbridge-Reitz normal distribution function
- G: Smith-GGX geometry term (separable form)
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
struct CookTorranceSpecularModel <: BRDFModel
    roughness::Float32
    F0::RGB{Float32}
end

"""Scalar convenience constructor (grey specular)."""
CookTorranceSpecularModel(roughness::Real, F0::Real) =
    CookTorranceSpecularModel(Float32(roughness), RGB{Float32}(F0, F0, F0))

# Microfacet helpers (_ggx_d, _smith_g1, _schlick) loaded from _microfacet_utils.jl

# ============================================================================
# Julia Evaluation
# ============================================================================

"""
    evaluate(model::CookTorranceSpecularModel, wi, wo) -> RGB{Float32}

Evaluate the pure-specular Cook-Torrance BRDF. Surface normal is (0, 0, 1).
"""
function evaluate(model::CookTorranceSpecularModel,
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

    denom = 4.0f0 * NdotL * NdotV
    return RGB{Float32}(
        max(0.0f0, D * G * Fr / denom),
        max(0.0f0, D * G * Fg / denom),
        max(0.0f0, D * G * Fb / denom)
    )
end

# ============================================================================
# GLSL Shader
# ============================================================================

function glsl_shader(::CookTorranceSpecularModel)::Tuple{String, Vector{String}}
    shader_path = joinpath(@__DIR__, "..", "rendering", "shaders", "cook_torrance_specular.glsl")
    code = read(shader_path, String)
    return (code, ["u_roughness", "u_F0"])
end

function shader_parameters(model::CookTorranceSpecularModel)::Dict{String, Any}
    return Dict{String, Any}(
        "u_roughness" => model.roughness,
        "u_F0"        => Float32[model.F0.r, model.F0.g, model.F0.b]
    )
end

# ============================================================================
# Registration
# ============================================================================

export CookTorranceSpecularModel
register_model!(CookTorranceSpecularModel)
