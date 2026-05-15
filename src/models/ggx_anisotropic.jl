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
- `albedo`:      Diffuse reflectance. Set to `RGB(0,0,0)` for pure specular.
"""
struct GGXAnisotropicModel <: BRDFModel
    roughness_x::Float32
    roughness_y::Float32
    F0::RGB{Float32}
    albedo::RGB{Float32}
end

"""Scalar convenience constructor (grey material)."""
function GGXAnisotropicModel(rx::Real, ry::Real, F0::Real, albedo::Real)
    GGXAnisotropicModel(
        Float32(rx),
        Float32(ry),
        RGB{Float32}(F0, F0, F0),
        RGB{Float32}(albedo, albedo, albedo),
    )
end

# ============================================================================
# Julia Evaluation
# ============================================================================

function evaluate(model::GGXAnisotropicModel,
                  wi::SVector{3, Float32},
                  wo::SVector{3, Float32})::RGB{Float32}
    NdotL = wi[3]
    NdotV = wo[3]
    if NdotL <= 0.0f0 || NdotV <= 0.0f0
        return RGB{Float32}(0, 0, 0)
    end

    h     = normalize(wi + wo)
    VdotH = clamp(dot(wo, h), 0.0f0, 1.0f0)

    ax = max(model.roughness_x * model.roughness_x, 1.0f-4)
    ay = max(model.roughness_y * model.roughness_y, 1.0f-4)

    D = _ggx_d_aniso(h[1], h[2], h[3], ax, ay)
    G = _smith_g1_aniso(wi[1], wi[2], wi[3], ax, ay) *
        _smith_g1_aniso(wo[1], wo[2], wo[3], ax, ay)

    Fr = _schlick(model.F0.r, VdotH)
    Fg = _schlick(model.F0.g, VdotH)
    Fb = _schlick(model.F0.b, VdotH)

    denom = 4.0f0 * NdotL * NdotV
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

function glsl_shader(::GGXAnisotropicModel)::Tuple{String, Vector{String}}
    shader_path = joinpath(@__DIR__, "..", "rendering", "shaders", "ggx_anisotropic.glsl")
    code = read(shader_path, String)
    uniforms = ["u_roughness_x", "u_roughness_y", "u_F0", "u_albedo"]
    return (code, uniforms)
end

function shader_parameters(model::GGXAnisotropicModel)::Dict{String, Any}
    return Dict{String, Any}(
        "u_roughness_x" => model.roughness_x,
        "u_roughness_y" => model.roughness_y,
        "u_F0"          => Float32[model.F0.r, model.F0.g, model.F0.b],
        "u_albedo"      => Float32[model.albedo.r, model.albedo.g, model.albedo.b],
    )
end

# ============================================================================
# Registration
# ============================================================================

export GGXAnisotropicModel
register_model!(GGXAnisotropicModel)
