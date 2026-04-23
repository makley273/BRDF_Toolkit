"""
Shared microfacet helper functions used by Cook-Torrance, GGX, and other
microfacet-based BRDF models.

These are loaded once and shared across all model plugins that need them.
"""

"""GGX/Trowbridge-Reitz normal distribution function. `alpha2` = α² (roughness⁴)."""
@inline function _ggx_d(NdotH::Float32, alpha2::Float32)::Float32
    d = NdotH * NdotH * (alpha2 - 1.0f0) + 1.0f0
    return alpha2 / (Float32(π) * d * d)
end

"""Smith-GGX masking function (single direction). `alpha2` = α²."""
@inline function _smith_g1(NdotV::Float32, alpha2::Float32)::Float32
    return 2.0f0 * NdotV / (NdotV + sqrt(alpha2 + (1.0f0 - alpha2) * NdotV * NdotV))
end

"""Schlick Fresnel approximation (scalar F0)."""
@inline function _schlick(F0::Float32, VdotH::Float32)::Float32
    x = 1.0f0 - VdotH
    return F0 + (1.0f0 - F0) * x * x * x * x * x
end
