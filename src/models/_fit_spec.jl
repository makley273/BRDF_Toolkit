"""
Shared fitting-spec vocabulary for BRDF model plugins.

A *fit spec* describes how a model's parameters map to/from a flat `Float64`
vector, plus the box bounds and the initial guess used by the optimizers. Every
optimizer (Levenberg-Marquardt, L-BFGS-B, ...) consumes the same spec, so a
model is fit identically regardless of the solver.

Each model owns its spec by defining a method of `_fit_spec` in its own file
(next to `evaluate`) — see `lambertian.jl` for the simplest example. This file
only provides the pieces shared across models:

- The bound constants (`_ROUGH_LO`, `_F0_LO`, ...) so per-model bounds stay in
  sync instead of drifting as literals.
- The generic fallback that errors for models without a spec.
- Factory helpers for the microfacet families (Cook-Torrance / GGX variants),
  whose parameter packing is identical across the two NDF choices.

This file sorts first in `models/` (leading underscore), so the vocabulary is
available by the time any model or optimizer is fit.

Returned NamedTuple: `(reconstruct, p0, lower, upper)`
  `reconstruct :: Vector{Float64} -> BRDFModel`
"""

# ============================================================================
# Parameter bounds shared across models
# ============================================================================

const _ROUGH_LO = 0.02
const _ROUGH_HI = 1.0
const _F0_LO    = 0.0
const _F0_HI    = 1.0
const _ALB_LO   = 0.0
const _ALB_HI   = 1.0
const _KD_LO    = 0.0
const _KD_HI    = 1.0

# ============================================================================
# Generic fallback
# ============================================================================

"""
    _fit_spec(::Type{<:BRDFModel}) -> (reconstruct, p0, lower, upper)

Return the parameter-packing spec for a model type. Each model defines its own
method in its plugin file; this fallback errors for models without one.
"""
_fit_spec(::Type{M}) where {M <: BRDFModel} =
    throw(ArgumentError("no fit spec defined for model type $M"))

# ============================================================================
# Microfacet family factories (shared by Cook-Torrance and GGX variants)
# ============================================================================

"""7-param microfacet + diffuse: (roughness, F0 rgb, albedo rgb)."""
function _spec_microfacet_diffuse(::Type{MT}) where {MT}
    return (
        reconstruct = p -> MT(p[1],
                              BRDFRGB(p[2], p[3], p[4]),
                              BRDFRGB(p[5], p[6], p[7])),
        p0    = [0.1,       0.04,   0.04,   0.04,     0.3,     0.3,     0.3],
        lower = [_ROUGH_LO, _F0_LO, _F0_LO, _F0_LO,   _ALB_LO, _ALB_LO, _ALB_LO],
        upper = [_ROUGH_HI, _F0_HI, _F0_HI, _F0_HI,   _ALB_HI, _ALB_HI, _ALB_HI],
    )
end

"""4-param pure-specular: (roughness, F0 rgb)."""
function _spec_microfacet_specular(::Type{MT}) where {MT}
    return (
        reconstruct = p -> MT(p[1], BRDFRGB(p[2], p[3], p[4])),
        p0    = [0.1,       0.04,   0.04,   0.04],
        lower = [_ROUGH_LO, _F0_LO, _F0_LO, _F0_LO],
        upper = [_ROUGH_HI, _F0_HI, _F0_HI, _F0_HI],
    )
end

"""8-param scaled-diffuse: (roughness, F0 rgb, albedo rgb, kd)."""
function _spec_microfacet_scaled(::Type{MT}) where {MT}
    return (
        reconstruct = p -> MT(p[1],
                              BRDFRGB(p[2], p[3], p[4]),
                              BRDFRGB(p[5], p[6], p[7]),
                              p[8]),
        p0    = [0.1,       0.04,   0.04,   0.04,     0.3,     0.3,     0.3,     0.5],
        lower = [_ROUGH_LO, _F0_LO, _F0_LO, _F0_LO,   _ALB_LO, _ALB_LO, _ALB_LO, _KD_LO],
        upper = [_ROUGH_HI, _F0_HI, _F0_HI, _F0_HI,   _ALB_HI, _ALB_HI, _ALB_HI, _KD_HI],
    )
end

"""8-param anisotropic + diffuse: (rough_x, rough_y, F0 rgb, albedo rgb)."""
function _spec_microfacet_aniso_diffuse(::Type{MT}) where {MT}
    return (
        reconstruct = p -> MT(p[1], p[2],
                              BRDFRGB(p[3], p[4], p[5]),
                              BRDFRGB(p[6], p[7], p[8])),
        p0    = [0.1,       0.1,       0.04,   0.04,   0.04,     0.3,     0.3,     0.3],
        lower = [_ROUGH_LO, _ROUGH_LO, _F0_LO, _F0_LO, _F0_LO,   _ALB_LO, _ALB_LO, _ALB_LO],
        upper = [_ROUGH_HI, _ROUGH_HI, _F0_HI, _F0_HI, _F0_HI,   _ALB_HI, _ALB_HI, _ALB_HI],
    )
end
