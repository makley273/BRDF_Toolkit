"""
CPU Sphere Renderer — deterministic Julia-side twin of the Three.js render path.

Renders the unit sphere with the SAME conventions as `renderer.js` +
`universal.glsl` so image-space metrics (SSIM) are computed on what the user
actually sees in the browser panels:

- Camera:  perspective, position (0,0,4), fov 45°, looking at the origin
           (`camera.position.set(0, 0, 4)` in renderer.js).
- Light:   point light at radius 30 on direction (sinθcosφ, sinθsinφ, cosθ)
           (`setLightDir` in renderer.js; effectively directional at r=30).
- Shading: `color = f(wi, wo) · NdotL · intensity` (universal.glsl line 93),
           then Reinhard tone mapping `c / (c + 1)` (renderer.js main()).
- NO gamma/sRGB conversion: the browser path is a raw ShaderMaterial writing
  gl_FragColor directly, so THREE's color-space pipeline never runs. We match.
- Local shading frame: world-up (0,1,0) tangent construction, identical to
  universal.glsl / _angles_to_dirs (only matters for anisotropic models).

One difference by design: the browser samples the float16 LUT while this
renderer evaluates `lookup`/`evaluate` directly — i.e. it renders the ground
truth the LUT approximates, which is the right reference for a fit metric.

Location: src/rendering/cpu_render.jl
"""

using StaticArrays
using Colors
using LinearAlgebra

# Scene constants mirrored from renderer.js.
const _CPU_CAM_POS      = SVector{3, Float32}(0.0f0, 0.0f0, 4.0f0)
const _CPU_CAM_FOV      = Float32(45.0 * π / 180.0)
const _CPU_LIGHT_RADIUS = 30.0f0
const _CPU_BACKGROUND   = RGB{Float32}(0x20 / 255, 0x20 / 255, 0x20 / 255)  # scene.background 0x202020

@inline _reinhard(x::Float32) = x / (x + 1.0f0)

"""
    render_brdf_sphere(f, light_theta, light_phi; resolution=256, intensity=π)
        -> (img::Matrix{RGB{Float32}}, mask::BitMatrix)

Ray-cast the unit sphere through the Three.js camera and shade each hit with
`f(wi, wo)::RGB` (directions in the LOCAL frame: z = surface normal), exactly
mirroring the browser pipeline (cosine term, `intensity` gain, Reinhard tone
mapping, no gamma). `mask[i,j]` is `true` where the ray hit the sphere;
non-hit pixels get the scene background color.

`light_theta`/`light_phi` follow the app's light-direction convention
(θ=0 → light on the camera axis).
"""
function render_brdf_sphere(f::Function, light_theta::Real, light_phi::Real;
                            resolution::Int = 256, intensity::Real = Float64(π))
    S = resolution
    img  = Matrix{RGB{Float32}}(undef, S, S)
    mask = falses(S, S)

    st, ct = sincos(Float32(light_theta))
    sp, cp = sincos(Float32(light_phi))
    light_pos = _CPU_LIGHT_RADIUS * SVector{3, Float32}(st * cp, st * sp, ct)

    tan_half = tan(_CPU_CAM_FOV / 2.0f0)
    C = _CPU_CAM_POS
    cc = dot(C, C) - 1.0f0                     # |C|² − r² for the unit sphere
    gain = Float32(intensity)

    @inbounds for i in 1:S, j in 1:S
        # Pixel center → NDC → camera ray (camera looks down −z, up = +y).
        ndc_x = (2.0f0 * (Float32(j) - 0.5f0) / S - 1.0f0) * tan_half
        ndc_y = (1.0f0 - 2.0f0 * (Float32(i) - 0.5f0) / S) * tan_half
        d = SVector{3, Float32}(ndc_x, ndc_y, -1.0f0)
        d = d / norm(d)

        # Ray-sphere: t² + 2t(C·d) + (|C|²−1) = 0; nearest root.
        b = dot(C, d)
        disc = b * b - cc
        if disc <= 0.0f0
            img[i, j] = _CPU_BACKGROUND
            continue
        end
        t = -b - sqrt(disc)
        P = C + t * d
        N = P                                   # unit sphere: normal = position
        mask[i, j] = true

        V = C - P; V = V / norm(V)
        L = light_pos - P; L = L / norm(L)

        # Local frame: world-up grain, matching universal.glsl / _angles_to_dirs.
        up = SVector{3, Float32}(0.0f0, 1.0f0, 0.0f0)
        T = cross(up, N)
        tl = norm(T)
        T = tl < 1.0f-5 ? SVector{3, Float32}(1.0f0, 0.0f0, 0.0f0) : T / tl
        Bt = cross(N, T)

        wi = SVector{3, Float32}(dot(L, T), dot(L, Bt), dot(L, N))
        wo = SVector{3, Float32}(dot(V, T), dot(V, Bt), dot(V, N))

        if wi[3] <= 0.0f0 || wo[3] <= 0.0f0
            img[i, j] = RGB{Float32}(0, 0, 0)   # shader returns vec3(0): unlit black
            continue
        end

        c = f(wi, wo)
        s = wi[3] * gain
        img[i, j] = RGB{Float32}(
            _reinhard(max(0.0f0, Float32(c.r) * s)),
            _reinhard(max(0.0f0, Float32(c.g) * s)),
            _reinhard(max(0.0f0, Float32(c.b) * s)),
        )
    end

    return img, mask
end

"""
    render_brdf_sphere(brdf::MeasuredBRDF, ...) / render_brdf_sphere(model::BRDFModel, ...)

Convenience wrappers shading with `lookup(brdf, wi, wo)` (measured data) or
`evaluate(model, wi, wo)` (fitted analytical model).
"""
render_brdf_sphere(brdf::MeasuredBRDF, light_theta::Real, light_phi::Real; kwargs...) =
    render_brdf_sphere((wi, wo) -> lookup(brdf, wi, wo), light_theta, light_phi; kwargs...)

render_brdf_sphere(model::BRDFModel, light_theta::Real, light_phi::Real; kwargs...) =
    render_brdf_sphere((wi, wo) -> evaluate(model, wi, wo), light_theta, light_phi; kwargs...)
