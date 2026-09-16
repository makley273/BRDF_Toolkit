"""
Universal 4D BRDF Lookup-Table Generator

Samples *any* `BRDFData` (measured data or a fitted analytical model) into a flat
4D lookup table over the Rusinkiewicz angles `(theta_h, theta_d, phi_d, phi_h)`,
ready to be uploaded as a folded 3D texture and sampled by `universal.glsl`.

The grid is folded into a 3D `Data3DTexture` so that the GPU's hardware trilinear
filter covers three of the four axes (`theta_h, theta_d, phi_d`); the shader adds
a manual interpolation across the folded fourth axis (`phi_h`, the anisotropy direction).

Layout (must stay in lock-step with `universal.glsl` and `renderer.js`):
- Texture dims: `(N_th, N_td, N_pd * N_ph)`  (width, height, depth)
- Depth fold:   `z = i_ph * N_pd + i_pd`
- Texel order:  `lin = i_th + N_th * (i_td + N_td * z)`  (x fastest — THREE.js order)
- Channels:     3 (RGB) on the wire; `renderer.js` expands to RGBA(alpha=1)

Angle warps are identical to the shader's forward mapping so the sampling is its
exact inverse:
- theta_h: sqrt warp (concentrates samples near the specular peak; cf. MERL)
- theta_d: linear over [0, π/2]
- phi_d:   linear over [0, 2π]  (full azimuth, endpoints inclusive — see below)
- phi_h:   linear over [0, 2π)  (periodic; anisotropy axis)

`phi_d` spans the WHOLE circle rather than a half-range. Halving it needs a
symmetry: reciprocity (`f(phi_d) == f(phi_d + π)`) or bilateral symmetry
(`f(phi_d) == f(-phi_d)`, ISOTROPIC materials only). Measured tables satisfy
neither exactly — folding by bilateral symmetry mirrors half the azimuth onto
the wrong data for anisotropic measurements (UTIA, RGL) and paints radial spokes
around the highlight, and folding by reciprocity seams non-reciprocal tables
(RGL). Storing the full circle assumes nothing. The range is closed at both ends
(`i_pd = N_pd - 1` is exactly 2π), so the first and last texel hold the *same*
configuration and the shader's clamped filter stays C0 across the wrap.

The angle box is over-complete: large `theta_h + theta_d` combinations reconstruct
to a pair with `wi` or `wo` below the horizon. Those cells are baked from the
nearest valid pair instead (`_max_theta_d`), so the shader's trilinear filter has
a sane value to blend against at grazing angles.

For isotropic data the value is independent of `phi_h`, so every `phi_h` block is
identical and the manual lerp is a no-op — but the 4D structure is preserved so the
grid size along `phi_h` can be tuned for anisotropy experiments.

Location: src/rendering/universal_texture.jl
"""

using StaticArrays
using LinearAlgebra
using Base64

# ── Forward warps (angle → continuous grid index), mirrored by universal.glsl ──

@inline _idx_theta_h(th::Float32, n::Int) =
    clamp(sqrt(max(0.0f0, th) / (Float32(π) / 2.0f0)) * Float32(n - 1), 0.0f0, Float32(n) - 1.0f0)

@inline _idx_linear(a::Float32, maxangle::Float32, n::Int) =
    clamp(a / maxangle * Float32(n - 1), 0.0f0, Float32(n) - 1.0f0)

# ── Inverse warps (integer texel index → angle), used while baking ─────────────

@inline _angle_theta_h(j::Int, n::Int) = (Float32(j) / Float32(n - 1))^2 * (Float32(π) / 2.0f0)
@inline _angle_linear(j::Int, maxangle::Float32, n::Int) = (Float32(j) / Float32(n - 1)) * maxangle

# ── Validity of a (theta_h, theta_d, phi_d) cell ───────────────────────────────

"""
    _max_theta_d(theta_h, phi_d; z_min) -> Float32

Largest `theta_d` for which `_angles_to_dirs(theta_h, ·, theta_d, phi_d)` still
returns a pair with both directions at least `z_min` above the horizon.

The Rusinkiewicz angles are over-complete: `(theta_h, theta_d, phi_d)` covers a
box, but only part of that box maps to a physically realizable reflection pair.
Writing the reconstruction in the frame around `H` (see `_angles_to_dirs`, where
`t.z = sin(theta_h)` and `b.z = 0`):

    wi.z, wo.z = cos(theta_h)cos(theta_d) ± sin(theta_h)cos(phi_d)sin(theta_d)

so with `A = cos(theta_h)`, `B = sin(theta_h)|cos(phi_d)|`,

    min(wi.z, wo.z) = A·cos(theta_d) − B·sin(theta_d) = R·cos(theta_d + φ)

for `R = hypot(A, B)` and `φ = atan(B, A)`. Requiring that to stay ≥ `z_min`
gives `theta_d ≤ acos(z_min / R) − φ`. Note `phi_d ∈ {π/2, 3π/2}` (`B = 0`) is
valid over the whole `theta_d` range, and `phi_d ∈ {0, π, 2π}` reduces to the
familiar `theta_h + theta_d ≤ π/2`. `|cos(phi_d)|` makes this hold over the full
`[0, 2π]` azimuth the bake loop walks.
"""
@inline function _max_theta_d(theta_h::Float32, phi_d::Float32;
                              z_min::Float32 = 1.0f-3)
    sth, cth = sincos(theta_h)
    A = cth
    B = sth * abs(cos(phi_d))
    R = sqrt(A * A + B * B)
    R < z_min && return 0.0f0            # theta_h ≈ π/2: h itself lies on the horizon
    return max(0.0f0, acos(clamp(z_min / R, -1.0f0, 1.0f0)) - atan(B, A))
end

"""
    _angles_to_dirs(theta_h, phi_h, theta_d, phi_d) -> (wi, wo)

Reconstruct incident/outgoing directions from the four Rusinkiewicz angles in the local frame `N = (0,0,1)`, `T = (1,0,0)`, `B = (0,1,0)`.

This is the exact inverse of the shader's forward mapping for the three
frame-independent axes (`theta_h, theta_d, phi_d`); `phi_h` sets the azimuth of the
half-vector in the tangent frame.
"""
function _angles_to_dirs(theta_h::Float32, phi_h::Float32,
                         theta_d::Float32, phi_d::Float32)
    N = SVector{3,Float32}(0.0f0, 0.0f0, 1.0f0)

    up = SVector{3,Float32}(0, 1, 0)
    T = cross(up, N)
    tl = norm(T)
    T = tl < 1.0f-5 ? SVector{3,Float32}(1, 0, 0) : T / tl
    Bt = cross(N, T)

    sth, cth = sincos(theta_h)
    sph, cph = sincos(phi_h)
    H = cth * N + sth * cph * T + sth * sph * Bt
    H = H / norm(H)

    # Frame around H: t = projection of N onto plane ⊥ H, b = H × t
    t = N - dot(N, H) * H
    tl = norm(t)
    if tl < 1.0f-6
        # H ∥ N (theta_h ≈ 0): pick an arbitrary tangent in the plane ⊥ H
        t = SVector{3,Float32}(1.0f0, 0.0f0, 0.0f0) - H[1] * H
        tl = norm(t)
    end
    t = t / tl
    b = cross(H, t)

    # Difference vector: polar theta_d from H, azimuth phi_d in (t, b)
    std, ctd = sincos(theta_d)
    spd, cpd = sincos(phi_d)
    wi = ctd * H + std * cpd * t + std * spd * b
    wi = wi / norm(wi)

    # wo is wi reflected about H
    wo = 2.0f0 * dot(wi, H) * H - wi
    wo = wo / norm(wo)

    return wi, wo
end

"""
    generate_lut(brdf; dims=(32, 32, 32, 32), dtype=:float16, intensity=100.0)
        -> (b64::String, config::Dict{String,Any})

Generate  a base64-encoded 4D LUT plus a JSON-friendly config describing
how `renderer.js` should upload it and how `universal.glsl` should sample it from a BRDF.

# Arguments
- `dims`:  `(N_theta_h, N_theta_d, N_phi_d, N_phi_h)`. Per-axis `8 ≤ N ≤ 128`
  (`N_phi_h` may be 1 for isotropic data). The grid is folded into a 3D texture,
  so additionally `N_phi_d × N_phi_h ≤ 2048` (the WebGL2 3D-texture depth limit).
- `dtype`: `:float16` (HalfFloatType) or `:float32` (FloatType).
- `intensity`: display gain applied in the shader (cosine-weighted). Default `π`
  turns a Lambertian BRDF (`albedo/π`) back into reflectance (`albedo`), so diffuse
  surfaces render at their true brightness and only real highlights clip.
"""
function generate_lut(brdf::BRDFData;
                            dims::NTuple{4,Int} = (32, 32, 32, 32),
                            dtype::Symbol = :float16,
                            intensity::Real = Float64(π))
    N_th, N_td, N_pd, N_ph = dims
    @assert all(8 .<= (N_th, N_td, N_pd) .<= 128) "theta_h/theta_d/phi_d dims must satisfy 8 ≤ N ≤ 128"
    @assert 1 <= N_ph <= 128 "phi_h dim must satisfy 1 ≤ N ≤ 128"
    @assert dtype in (:float16, :float32) "dtype must be :float16 or :float32"
    # The 4D grid is folded into a 3D texture whose depth is N_phi_d × N_phi_h.
    # WebGL2 guarantees MAX_3D_TEXTURE_SIZE ≥ 2048 (and most GPUs report exactly
    # 2048), so a larger folded depth fails to upload and renders black. Reject it
    # here rather than let it surface as an opaque GPU error in the browser.
    @assert N_pd * N_ph <= 2048 "folded texture depth N_phi_d×N_phi_h = $(N_pd * N_ph) exceeds the GPU 3D-texture limit (2048); reduce phi_d or phi_h"

    n_texels = N_th * N_td * N_pd * N_ph
    rgb = Vector{Float32}(undef, n_texels * 3)

    half = Float32(π) / 2.0f0
    @inbounds for i_ph in 0:(N_ph - 1)
        phi_h = (Float32(i_ph) / Float32(N_ph)) * (2.0f0 * Float32(π))   # periodic [0,2π)
        for i_pd in 0:(N_pd - 1)
            phi_d = _angle_linear(i_pd, 2.0f0 * Float32(π), N_pd)        # [0,2π]
            for i_td in 0:(N_td - 1)
                theta_d_grid = _angle_linear(i_td, half, N_td)
                for i_th in 0:(N_th - 1)
                    theta_h = _angle_theta_h(i_th, N_th)

                    # Cells past the horizon hold no realizable direction pair.
                    # Bake the nearest valid pair (same theta_h/phi_d, theta_d
                    # pulled back onto the boundary) instead of querying an
                    # impossible geometry: analytical models answer 0 there while
                    # measured tables extrapolate garbage, and either way the
                    # shader's trilinear filter drags that value into neighbouring
                    # valid cells at grazing angles.
                    theta_d = min(theta_d_grid, _max_theta_d(theta_h, phi_d))

                    wi, wo = _angles_to_dirs(theta_h, phi_h, theta_d, phi_d)

                    # theta_h = π/2 puts h itself on the horizon, so no theta_d
                    # yields a realizable pair and there is nothing to clamp to.
                    # Zero is the physical answer for that slice (and guards the
                    # invariant against roundoff, cos(Float32(π)/2) ≈ -4e-8).
                    c = (wi[3] > 0.0f0 && wo[3] > 0.0f0) ?
                        lookup(brdf, wi, wo) : RGB{Float32}(0, 0, 0)

                    z = i_ph * N_pd + i_pd
                    lin = i_th + N_th * (i_td + N_td * z)          # 0-based texel
                    o = lin * 3
                    rgb[o + 1] = max(0.0f0, Float32(c.r))
                    rgb[o + 2] = max(0.0f0, Float32(c.g))
                    rgb[o + 3] = max(0.0f0, Float32(c.b))
                end
            end
        end
    end

    # Encode to the requested precision (little-endian; matches browser/THREE).
    b64 = if dtype == :float16
        base64encode(reinterpret(UInt8, Float16.(rgb)))
    else
        base64encode(reinterpret(UInt8, rgb))
    end

    config = Dict{String,Any}(
        "format"    => "universal",
        "dims"      => [N_th, N_td, N_pd, N_ph],
        "dtype"     => String(dtype),
        "intensity" => Float64(intensity),
    )

    return b64, config
end
