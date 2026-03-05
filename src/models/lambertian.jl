"""
Lambertian (ideal diffuse) BRDF model.

The Lambertian BRDF is constant across all directions:
    f(wi, wo) = albedo / π

It serves as the diffuse baseline and the simplest possible BRDFModel plugin.
"""

using StaticArrays
using Colors

# ============================================================================
# Model Type
# ============================================================================

"""
    LambertianModel <: BRDFModel

Ideal diffuse BRDF. Single RGB albedo parameter.

# Fields
- `albedo`: Diffuse reflectance per channel ∈ [0, 1]³
"""
struct LambertianModel <: BRDFModel
    albedo::RGB{Float32}
end

"""Convenience constructor: scalar albedo applied to all channels."""
LambertianModel(albedo::Real) = LambertianModel(RGB{Float32}(albedo, albedo, albedo))

# ============================================================================
# Julia Evaluation
# ============================================================================

"""
    evaluate(model::LambertianModel, wi, wo) -> RGB{Float32}

Returns `albedo / π` for directions in the upper hemisphere, zero otherwise.
"""
function evaluate(model::LambertianModel,
                  wi::SVector{3, Float32},
                  wo::SVector{3, Float32})::RGB{Float32}
    if wi[3] <= 0.0f0 || wo[3] <= 0.0f0
        return RGB{Float32}(0, 0, 0)
    end
    c = Float32(1.0 / π)
    return RGB{Float32}(model.albedo.r * c, model.albedo.g * c, model.albedo.b * c)
end

# ============================================================================
# GLSL Shader
# ============================================================================

function glsl_shader(::LambertianModel)::Tuple{String, Vector{String}}
    shader_path = joinpath(@__DIR__, "..", "rendering", "shaders", "lambertian.glsl")
    code = read(shader_path, String)
    return (code, ["u_albedo"])
end

function shader_parameters(model::LambertianModel)::Dict{String, Any}
    return Dict{String, Any}(
        "u_albedo" => Float32[model.albedo.r, model.albedo.g, model.albedo.b]
    )
end

# ============================================================================
# Registration
# ============================================================================

export LambertianModel
register_model!(LambertianModel)
