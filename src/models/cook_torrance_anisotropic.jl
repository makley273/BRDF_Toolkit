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
- `albedo`:      Diffuse reflectance. Set to `RGB(0,0,0)` for pure specular.
"""
struct CookTorranceAnisotropicModel <: BRDFModel
    roughness_x::Float32
    roughness_y::Float32
    F0::RGB{Float32}
    albedo::RGB{Float32}
end

"""Scalar convenience constructor (grey material)."""
function CookTorranceAnisotropicModel(rx::Real, ry::Real, F0::Real, albedo::Real)
    CookTorranceAnisotropicModel(
        Float32(rx),
        Float32(ry),
        RGB{Float32}(F0, F0, F0),
        RGB{Float32}(albedo, albedo, albedo),
    )
end

# ============================================================================
# Julia Evaluation
# ============================================================================

function evaluate(model::CookTorranceAnisotropicModel,
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

    ax = max(model.roughness_x * model.roughness_x, 1.0f-4)
    ay = max(model.roughness_y * model.roughness_y, 1.0f-4)

    D = _beckmann_d_aniso(h[1], h[2], h[3], ax, ay)
    G = _vcavity_g(NdotL, NdotV, NdotH, VdotH)

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

function glsl_shader(::CookTorranceAnisotropicModel)::Tuple{String, Vector{String}}
    shader_path = joinpath(@__DIR__, "..", "rendering", "shaders", "cook_torrance_anisotropic.glsl")
    code = read(shader_path, String)
    uniforms = ["u_roughness_x", "u_roughness_y", "u_F0", "u_albedo"]
    return (code, uniforms)
end

function shader_parameters(model::CookTorranceAnisotropicModel)::Dict{String, Any}
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

export CookTorranceAnisotropicModel
register_model!(CookTorranceAnisotropicModel)
