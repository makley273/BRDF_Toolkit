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
struct LambertianModel{T <: Real} <: BRDFModel
    albedo::BRDFRGB{T}
end

"""Convenience constructor: scalar albedo applied to all channels."""
LambertianModel(albedo::Real) = LambertianModel(BRDFRGB(float(albedo), float(albedo), float(albedo)))

# ============================================================================
# Julia Evaluation
# ============================================================================

"""
    evaluate(model::LambertianModel, wi, wo)

Returns `albedo / π` for directions in the upper hemisphere, zero otherwise.
"""
function evaluate(model::LambertianModel, wi, wo)
    if wi[3] <= 0.0 || wo[3] <= 0.0
        z = zero(model.albedo.r)
        return BRDFRGB(z, z, z)
    end
    c = 1.0 / π
    return BRDFRGB(model.albedo.r * c, model.albedo.g * c, model.albedo.b * c)
end

# ============================================================================
# Fit Spec
# ============================================================================

# 3 params: albedo rgb ∈ [0, 1]³
_fit_spec(::Type{LambertianModel}) = (
    reconstruct = p -> LambertianModel(BRDFRGB(p[1], p[2], p[3])),
    p0    = [0.5, 0.5, 0.5],
    lower = [_ALB_LO, _ALB_LO, _ALB_LO],
    upper = [_ALB_HI, _ALB_HI, _ALB_HI],
)

# ============================================================================
# Registration
# ============================================================================

register_model!(LambertianModel)
