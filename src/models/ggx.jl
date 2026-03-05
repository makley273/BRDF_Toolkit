"""
GGX pure-specular microfacet BRDF model.

Implements the specular-only Cook-Torrance framework with GGX components:
    f(wi, wo) = D(h) · G(wi,wo) · F(wi,h) / (4 · NdotL · NdotV)

Unlike CookTorranceModel there is no diffuse term — this model represents
metallic or mirror-like surfaces where all reflectance is specular.

Components (identical to CookTorranceModel):
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

Pure-specular microfacet BRDF. No diffuse component.

# Fields
- `roughness`: Perceptual roughness ∈ [0, 1]. α = roughness² used internally.
- `F0`:        Fresnel reflectance at normal incidence (specular colour).
"""
struct GGXModel <: BRDFModel
    roughness::Float32
    F0::RGB{Float32}
end

"""Scalar convenience constructor (grey specular)."""
GGXModel(roughness::Real, F0::Real) = GGXModel(Float32(roughness), RGB{Float32}(F0, F0, F0))

# ============================================================================
# Internal microfacet helpers
# (Identical to those in cook_torrance.jl — redefined here for self-containment.
#  If cook_torrance.jl was already loaded, Julia silently overwrites with the
#  same method, which is harmless.)
# ============================================================================

@inline function _ggx_d(NdotH::Float32, alpha2::Float32)::Float32
    d = NdotH * NdotH * (alpha2 - 1.0f0) + 1.0f0
    return alpha2 / (Float32(π) * d * d)
end

@inline function _smith_g1(NdotV::Float32, alpha2::Float32)::Float32
    return 2.0f0 * NdotV / (NdotV + sqrt(alpha2 + (1.0f0 - alpha2) * NdotV * NdotV))
end

@inline function _schlick(F0::Float32, VdotH::Float32)::Float32
    x = 1.0f0 - VdotH
    return F0 + (1.0f0 - F0) * x * x * x * x * x
end

# ============================================================================
# Julia Evaluation
# ============================================================================

"""
    evaluate(model::GGXModel, wi, wo) -> RGB{Float32}

Pure specular GGX BRDF. Surface normal is (0, 0, 1).
"""
function evaluate(model::GGXModel,
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

function glsl_shader(::GGXModel)::Tuple{String, Vector{String}}
    shader_path = joinpath(@__DIR__, "..", "rendering", "shaders", "ggx.glsl")
    code = read(shader_path, String)
    return (code, ["u_roughness", "u_F0"])
end

function shader_parameters(model::GGXModel)::Dict{String, Any}
    return Dict{String, Any}(
        "u_roughness" => model.roughness,
        "u_F0"        => Float32[model.F0.r, model.F0.g, model.F0.b]
    )
end

# ============================================================================
# Registration
# ============================================================================

export GGXModel
register_model!(GGXModel)
