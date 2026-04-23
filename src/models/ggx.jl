"""
GGX microfacet BRDF model with optional Lambertian diffuse component.

    f(wi, wo) = albedo/π  +  D(h) · G(wi,wo) · F(wi,h) / (4 · NdotL · NdotV)

Set albedo = RGB(0,0,0) for a pure-specular (metallic) surface.

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
    GGXModel <: BRDFModel

GGX microfacet BRDF with Lambertian diffuse component.

# Fields
- `roughness`: Perceptual roughness ∈ [0, 1]. α = roughness² used internally.
- `F0`:        Fresnel reflectance at normal incidence (specular colour).
- `albedo`:    Diffuse reflectance. Use RGB(0,0,0) for pure-specular (metallic) surfaces.
"""
struct GGXModel <: BRDFModel
    roughness::Float32
    F0::RGB{Float32}
    albedo::RGB{Float32}
end

"""Scalar convenience constructor (grey specular, no diffuse)."""
GGXModel(roughness::Real, F0::Real) =
    GGXModel(Float32(roughness), RGB{Float32}(F0, F0, F0), RGB{Float32}(0, 0, 0))

"""Scalar convenience constructor with grey diffuse."""
GGXModel(roughness::Real, F0::Real, albedo::Real) =
    GGXModel(Float32(roughness), RGB{Float32}(F0, F0, F0), RGB{Float32}(albedo, albedo, albedo))

# Microfacet helpers (_ggx_d, _smith_g1, _schlick) loaded from _microfacet_utils.jl

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
    spec_r = D * G * Fr / denom
    spec_g = D * G * Fg / denom
    spec_b = D * G * Fb / denom

    inv_pi = 1.0f0 / Float32(π)
    return RGB{Float32}(
        max(0.0f0, model.albedo.r * inv_pi + spec_r),
        max(0.0f0, model.albedo.g * inv_pi + spec_g),
        max(0.0f0, model.albedo.b * inv_pi + spec_b)
    )
end

# ============================================================================
# GLSL Shader
# ============================================================================

function glsl_shader(::GGXModel)::Tuple{String, Vector{String}}
    shader_path = joinpath(@__DIR__, "..", "rendering", "shaders", "ggx.glsl")
    code = read(shader_path, String)
    return (code, ["u_roughness", "u_F0", "u_albedo"])
end

function shader_parameters(model::GGXModel)::Dict{String, Any}
    return Dict{String, Any}(
        "u_roughness" => model.roughness,
        "u_F0"        => Float32[model.F0.r, model.F0.g, model.F0.b],
        "u_albedo"    => Float32[model.albedo.r, model.albedo.g, model.albedo.b]
    )
end

# ============================================================================
# Registration
# ============================================================================

export GGXModel
register_model!(GGXModel)
