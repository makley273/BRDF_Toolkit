"""
Ashikhmin-Shirley anisotropic Phong BRDF model.

    f(wi, wo) = ρs(wi, wo) + ρd(wi, wo)

A physically plausible, energy-conserving Phong variant (Ashikhmin & Shirley,
"An Anisotropic Phong BRDF Model", JGT 2000). Unlike the microfacet family
(Cook-Torrance / GGX), the specular lobe is an *anisotropic Phong* cosine power
about the half-vector, and the diffuse term is a non-Lambertian, Fresnel-coupled
expression that keeps the total reflectance ≤ 1.

Specular term:
    ρs = √((nu+1)(nv+1)) / (8π)
         · (n·h)^p / ( (h·wi) · max(n·wi, n·wo) )
         · F(h·wi)

    with anisotropic exponent
        p = (nu·(h·u)² + nv·(h·v)²) / (1 − (h·n)²)

- nu, nv: Phong exponents along the surface tangent u (+x) and bitangent v (+y).
  Equal exponents give an isotropic highlight; unequal ones stretch it into an
  elongated lobe (brushed metal, hair, fabric).
- F: Schlick Fresnel about the specular reflectance Rs at normal incidence.

Diffuse term (energy-coupled to the specular via (1 − Rs)):
    ρd = (28·Rd)/(23π) · (1 − Rs)
         · (1 − (1 − (n·wi)/2)^5)
         · (1 − (1 − (n·wo)/2)^5)

- Rd: diffuse reflectance. Rs: specular reflectance at normal incidence.

Directions arrive in the shading frame (normal = +z, tangent u = +x, bitangent
v = +y), so the half-vector components (h[1], h[2], h[3]) are the u / v / n
projections directly — matching `cook_torrance_anisotropic.jl`.

Reference:
- Ashikhmin & Shirley, "An Anisotropic Phong BRDF Model", JGT 5(2), 2000.
"""

using StaticArrays
using Colors
using LinearAlgebra

# ============================================================================
# Model Type
# ============================================================================

"""
    AshikhminShirleyModel <: BRDFModel

Anisotropic Phong BRDF: anisotropic Phong specular lobe + Fresnel-coupled,
energy-conserving diffuse term.

# Fields
- `nu`: Phong exponent along the tangent (+x). Larger ⇒ sharper highlight.
- `nv`: Phong exponent along the bitangent (+y). `nu == nv` ⇒ isotropic.
- `Rs`: Specular reflectance at normal incidence (Fresnel F0).
- `Rd`: Diffuse reflectance. Set to `BRDFRGB(0,0,0)` for a pure specular material.
"""
struct AshikhminShirleyModel{T <: Real} <: BRDFModel
    nu::T
    nv::T
    Rs::BRDFRGB{T}
    Rd::BRDFRGB{T}
end

"""Scalar convenience constructor (grey material)."""
function AshikhminShirleyModel(nu::Real, nv::Real, Rs::Real, Rd::Real)
    T = float(promote_type(typeof(nu), typeof(nv), typeof(Rs), typeof(Rd)))
    AshikhminShirleyModel(T(nu), T(nv), BRDFRGB{T}(Rs, Rs, Rs), BRDFRGB{T}(Rd, Rd, Rd))
end

# ============================================================================
# Julia Evaluation
# ============================================================================

function evaluate(model::AshikhminShirleyModel, wi, wo)
    NdotL = wi[3]
    NdotV = wo[3]
    if NdotL <= 0.0 || NdotV <= 0.0
        z = zero(model.nu)
        return BRDFRGB(z, z, z)
    end

    h     = normalize(wi + wo)
    NdotH = clamp(h[3], 0.0, 1.0)
    HdotL = max(dot(wi, h), 1e-6)          # h·wi == h·wo (h bisects wi, wo)

    nu = model.nu
    nv = model.nv

    # Anisotropic Phong exponent p = (nu·hx² + nv·hy²) / (1 − hz²).
    # 1 − hz² == hx² + hy² for a unit h; floor it so h ≈ n (highlight centre)
    # stays finite. There the base (n·h) → 1, so (n·h)^p → 1 regardless of p.
    denom_uv = max(h[1] * h[1] + h[2] * h[2], 1e-7)
    p = (nu * h[1] * h[1] + nv * h[2] * h[2]) / denom_uv

    spec_norm = sqrt((nu + 1.0) * (nv + 1.0)) / (8.0 * π)
    spec_geom = NdotH^p / (HdotL * max(NdotL, NdotV))
    spec_base = spec_norm * spec_geom      # per-channel factor before Fresnel

    # Fresnel-coupled diffuse: 28/(23π)·(1−Rs)·(1−(1−n·wi/2)^5)·(1−(1−n·wo/2)^5).
    fl = 1.0 - _pow5(1.0 - NdotL / 2.0)
    fv = 1.0 - _pow5(1.0 - NdotV / 2.0)
    diff_base = (28.0 / (23.0 * π)) * fl * fv

    Fr = _schlick(model.Rs.r, HdotL)
    Fg = _schlick(model.Rs.g, HdotL)
    Fb = _schlick(model.Rs.b, HdotL)

    r = spec_base * Fr + model.Rd.r * (1.0 - model.Rs.r) * diff_base
    g = spec_base * Fg + model.Rd.g * (1.0 - model.Rs.g) * diff_base
    b = spec_base * Fb + model.Rd.b * (1.0 - model.Rs.b) * diff_base

    return BRDFRGB(max(zero(r), r), max(zero(g), g), max(zero(b), b))
end

"""Fifth power via repeated multiplication (ForwardDiff-friendly, no `^`)."""
@inline _pow5(x) = (x * x * x * x * x)

# ============================================================================
# Fit Spec
# ============================================================================

# 8 params: (nu, nv, Rs rgb, Rd rgb). The Phong exponents span a wide, highly
# nonlinear range; [1, 1000] covers rough-to-mirror without letting the solver
# wander to degenerate values. Rs / Rd are reflectances, reusing [0, 1] bounds.
const _EXP_LO = 1.0
const _EXP_HI = 1000.0

_fit_spec(::Type{AshikhminShirleyModel}) = (
    reconstruct = p -> AshikhminShirleyModel(p[1], p[2],
                                             BRDFRGB(p[3], p[4], p[5]),
                                             BRDFRGB(p[6], p[7], p[8])),
    p0    = [100.0,   100.0,   0.05,   0.05,   0.05,     0.3,     0.3,     0.3],
    lower = [_EXP_LO, _EXP_LO, _F0_LO, _F0_LO, _F0_LO,   _ALB_LO, _ALB_LO, _ALB_LO],
    upper = [_EXP_HI, _EXP_HI, _F0_HI, _F0_HI, _F0_HI,   _ALB_HI, _ALB_HI, _ALB_HI],
)

# ============================================================================
# Registration
# ============================================================================

register_model!(AshikhminShirleyModel)
