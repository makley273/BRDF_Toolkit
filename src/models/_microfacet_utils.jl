"""
Shared microfacet helper functions used by Cook-Torrance, GGX, and other
microfacet-based BRDF models.

Two distinct microfacet families are provided:

  Cook-Torrance (1982): Beckmann NDF + V-cavity geometry term.
    - Beckmann: Gaussian-on-slopes distribution; narrower / sharper highlight.
    - V-cavity G: geometric occlusion based on symmetric V-grooves, derived
      independently of the slope distribution.

  GGX / Trowbridge-Reitz (1975; Walter et al. 2007): GGX NDF + Smith masking.
    - GGX: power-law tails; broader / longer highlights than Beckmann.
    - Smith G: derived consistently from the slope distribution.

Anisotropic forms use separate roughnesses αx, αy along tangent / bitangent.
In the shading frame the surface normal is (0,0,1); halfvector components
(hx, hy, hz) give the tangent / bitangent / normal projections directly.

Roughness parameterisation throughout: α = roughness² (Disney/Unreal convention).
"""

# ============================================================================
# GGX family (isotropic)
# ============================================================================

"""GGX/Trowbridge-Reitz normal distribution function. `alpha2` = α² (roughness⁴)."""
@inline function _ggx_d(NdotH::Float32, alpha2::Float32)::Float32
    d = NdotH * NdotH * (alpha2 - 1.0f0) + 1.0f0
    return alpha2 / (Float32(π) * d * d)
end

"""Smith-GGX masking function (single direction). `alpha2` = α²."""
@inline function _smith_g1(NdotV::Float32, alpha2::Float32)::Float32
    return 2.0f0 * NdotV / (NdotV + sqrt(alpha2 + (1.0f0 - alpha2) * NdotV * NdotV))
end

# ============================================================================
# Cook-Torrance family (isotropic)
# ============================================================================

"""Beckmann normal distribution function. `alpha2` = α²."""
@inline function _beckmann_d(NdotH::Float32, alpha2::Float32)::Float32
    NdotH2 = max(NdotH * NdotH, 1.0f-6)
    tan2   = (1.0f0 - NdotH2) / NdotH2
    return exp(-tan2 / alpha2) / (Float32(π) * alpha2 * NdotH2 * NdotH2)
end

"""
Cook-Torrance V-cavity geometric attenuation. Independent of D.
Returns the combined G for both incoming and outgoing directions.
"""
@inline function _vcavity_g(NdotL::Float32, NdotV::Float32,
                             NdotH::Float32, VdotH::Float32)::Float32
    vh = max(VdotH, 1.0f-6)
    a = 2.0f0 * NdotH * NdotV / vh
    b = 2.0f0 * NdotH * NdotL / vh
    return min(1.0f0, min(a, b))
end

# ============================================================================
# Anisotropic forms
# ============================================================================

"""
Anisotropic GGX NDF. `h` is the halfvector in the shading frame
(tangent = +x, bitangent = +y, normal = +z).
"""
@inline function _ggx_d_aniso(hx::Float32, hy::Float32, hz::Float32,
                               ax::Float32, ay::Float32)::Float32
    ax_s = max(ax, 1.0f-4); ay_s = max(ay, 1.0f-4)
    r = (hx / ax_s) * (hx / ax_s) + (hy / ay_s) * (hy / ay_s) + hz * hz
    return 1.0f0 / (Float32(π) * ax_s * ay_s * r * r)
end

"""Anisotropic Smith-GGX masking (single direction)."""
@inline function _smith_g1_aniso(vx::Float32, vy::Float32, vz::Float32,
                                  ax::Float32, ay::Float32)::Float32
    vz_s = max(vz, 1.0f-6)
    lam_arg = (ax * ax * vx * vx + ay * ay * vy * vy) / (vz_s * vz_s)
    lam = 0.5f0 * (-1.0f0 + sqrt(1.0f0 + lam_arg))
    return 1.0f0 / (1.0f0 + lam)
end

"""Anisotropic Beckmann NDF in the shading frame."""
@inline function _beckmann_d_aniso(hx::Float32, hy::Float32, hz::Float32,
                                    ax::Float32, ay::Float32)::Float32
    ax_s = max(ax, 1.0f-4); ay_s = max(ay, 1.0f-4)
    hz2  = max(hz * hz, 1.0f-6)
    exponent = -((hx * hx) / (ax_s * ax_s) + (hy * hy) / (ay_s * ay_s)) / hz2
    return exp(exponent) / (Float32(π) * ax_s * ay_s * hz2 * hz2)
end

# ============================================================================
# Fresnel
# ============================================================================

"""Schlick Fresnel approximation (scalar F0)."""
@inline function _schlick(F0::Float32, VdotH::Float32)::Float32
    x = 1.0f0 - VdotH
    return F0 + (1.0f0 - F0) * x * x * x * x * x
end
