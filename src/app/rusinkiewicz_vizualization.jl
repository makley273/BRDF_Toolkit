"""
Rusinkiewicz parametrization vizuallization 
"""

using Bonito
using WGLMakie
using Base64
using LinearAlgebra: dot, normalize
import WGLMakie.GeometryBasics as GeometryBasics
import CairoMakie
import JpegTurbo
using Colors: RGB

# Loading a Makie backend activates it; the app must stay on WGLMakie.
WGLMakie.activate!()

# ============================================================================
# Coordinate conversions (self-contained; conventions match src/formats/merl.jl)
# ============================================================================

_sph_dir(θ::Real, φ::Real) = Vec3d(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))

"""
    _rot_hframe(θh, φh, v) -> Vec3d

Rotate `v` from the half-vector frame into world coordinates: Rz(φh)·Ry(θh)·v.
The frame's pole (0,0,1) maps onto h.
"""
function _rot_hframe(θh::Real, φh::Real, v)
    x1 = cos(θh) * v[1] + sin(θh) * v[3]
    y1 = v[2]
    z1 = -sin(θh) * v[1] + cos(θh) * v[3]
    return Vec3d(cos(φh) * x1 - sin(φh) * y1,
                 sin(φh) * x1 + cos(φh) * y1,
                 z1)
end

"""
    rusinkiewicz_to_directions(θh, φh, θd, φd) -> (wi, wo, h)

Half/difference angles → direction pair. `wo` may dip below the horizon for
large θh + θd combinations (not every coordinate tuple is a valid reflection).
"""
function rusinkiewicz_to_directions(θh::Real, φh::Real, θd::Real, φd::Real)
    h  = _sph_dir(θh, φh)
    wi = _rot_hframe(θh, φh, _sph_dir(θd, φd))
    wo = 2 * dot(wi, h) * h - wi
    return wi, wo, h
end

"""
    directions_to_rusinkiewicz(wi, wo) -> (θh, φh, θd, φd)

Direction pair → half/difference angles. φh and φd are returned in [0, 2π);
both degenerate cases (h ∥ n, d ∥ h) collapse the respective azimuth to 0.
"""
function directions_to_rusinkiewicz(wi, wo)
    h  = normalize(Vec3d(wi) + Vec3d(wo))
    θh = acos(clamp(h[3], -1.0, 1.0))
    φh = (abs(h[1]) < 1e-9 && abs(h[2]) < 1e-9) ? 0.0 : mod2pi(atan(h[2], h[1]))
    # d = Ry(-θh)·Rz(-φh)·wi
    x1 = cos(φh) * wi[1] + sin(φh) * wi[2]
    y1 = -sin(φh) * wi[1] + cos(φh) * wi[2]
    z1 = wi[3]
    d  = Vec3d(cos(θh) * x1 - sin(θh) * z1, y1, sin(θh) * x1 + cos(θh) * z1)
    θd = acos(clamp(d[3], -1.0, 1.0))
    φd = (abs(d[1]) < 1e-9 && abs(d[2]) < 1e-9) ? 0.0 : mod2pi(atan(d[2], d[1]))
    return θh, φh, θd, φd
end

const _RUS_COL = (
    wi     = RGBf(0.20, 0.60, 0.86),   # blue   
    wo     = RGBf(0.15, 0.68, 0.38),   # green  
    h      = RGBf(0.90, 0.49, 0.13),   # orange 
    n      = RGBf(0.17, 0.24, 0.31),   # dark   
    guide  = RGBf(0.55, 0.57, 0.58),   # gray
)

function _slerp(a, b, t::Real)
    an = normalize(Vec3d(a)); bn = normalize(Vec3d(b))
    ω = acos(clamp(dot(an, bn), -1.0, 1.0))
    ω < 1e-6 && return an
    return (sin((1 - t) * ω) * an + sin(t * ω) * bn) / sin(ω)
end

_arc_points(a, b, r; n::Int = 33) =
    Point3f[Point3f(r * _slerp(a, b, i / (n - 1))) for i in 0:(n - 1)]

"""
    _construction_axis!(figpos; azimuth, elevation) -> Axis3

Bare Axis3 (no decorations, fixed limits) for one construction panel. Both
panels share the same limits, so the two parameterizations sit in an identical
world and can be read against each other.
"""
function _construction_axis!(figpos; azimuth::Real = 0.55, elevation::Real = π / 9)
    ax = Axis3(figpos; aspect = :data, azimuth = azimuth, elevation = elevation,
               viewmode = :stretch, protrusions = 0)
    hidedecorations!(ax)
    hidespines!(ax)
    ax.limits[] = ((-1.22, 1.22), (-1.22, 1.22), (-0.02, 1.20))
    return ax
end

# Explicit data-space arrow sizing - the auto markerscale is far too chunky for
# a handful of unit vectors in one scene.
const _RUS_ARROW = (markerscale = 1.0, shaftradius = 0.010,
                    tipradius = 0.030, tiplength = 0.09, shading = NoShading)
const _RUS_THIN_ARROW = (markerscale = 1.0, shaftradius = 0.005,
                         tipradius = 0.018, tiplength = 0.06, shading = NoShading)

_ground_arc(φ::Real, r::Real; n::Int = 49) =
    Point3f[Point3f(r * cos(t), r * sin(t), 0) for t in range(0, φ; length = n)]

"""
    _draw_stage!(ax)

Static scenery shared by both panels: ground disk with horizon circle, the
x/y axes and the surface normal n.
"""
function _draw_stage!(ax)
    circle = Point3f[Point3f(cos(φ), sin(φ), 0) for φ in range(0, 2π; length = 129)]
    disk_verts = vcat([Point3f(0, 0, 0)], circle)
    disk_faces = [GeometryBasics.TriangleFace(1, i, i + 1) for i in 2:129]
    mesh!(ax, disk_verts, disk_faces;
          color = (:gray, 0.08), shading = NoShading, transparency = true)
    lines!(ax, circle; color = (_RUS_COL.guide, 0.6), linewidth = 1)

    origin = [Point3f(0, 0, 0)]
    arrows3d!(ax, origin, [Vec3f(1.08, 0, 0)];
              color = (_RUS_COL.n, 0.7), _RUS_THIN_ARROW...)
    arrows3d!(ax, origin, [Vec3f(0, 1.08, 0)];
              color = (_RUS_COL.n, 0.7), _RUS_THIN_ARROW...)
    arrows3d!(ax, origin, [Vec3f(0, 0, 1)]; color = _RUS_COL.n, _RUS_ARROW...)
    text!(ax, Point3f(1.18, 0, 0); text = L"x", color = _RUS_COL.n,
          fontsize = 15, align = (:center, :center))
    text!(ax, Point3f(0, 1.18, 0); text = L"y", color = _RUS_COL.n,
          fontsize = 15, align = (:center, :center))
    text!(ax, Point3f(-0.13, 0, 1.10); text = L"n", color = _RUS_COL.n,
          fontsize = 18, align = (:center, :center))
    return ax
end

"""
    draw_spherical_construction!(ax, state)

Draw the standard parameterization into `ax`: ωᵢ and ωₒ with the polar angles
θᵢ/θₒ measured from n and the azimuths φᵢ/φₒ measured from the x axis in the
ground plane. `state` is the same `(th, ph, td, pd)` NamedTuple in **radians**
(or an `Observable` of one) that drives the Rusinkiewicz panel, so both panels
always show the very same direction pair.
"""
function draw_spherical_construction!(ax, state)
    state isa Observable || (state = Observable(state))
    _draw_stage!(ax)

    geo = lift(state) do s
        wi, wo, _ = rusinkiewicz_to_directions(s.th, s.ph, s.td, s.pd)
        n  = Vec3d(0, 0, 1)
        θi = acos(clamp(wi[3], -1.0, 1.0)); φi = mod2pi(atan(wi[2], wi[1]))
        θo = acos(clamp(wo[3], -1.0, 1.0)); φo = mod2pi(atan(wo[2], wo[1]))
        (
            wi = Vec3f(wi), wo = Vec3f(wo),
            ti_arc = _arc_points(n, wi, 0.50),
            to_arc = _arc_points(n, wo, 0.64),
            pi_arc = _ground_arc(φi, 0.32),
            po_arc = _ground_arc(φo, 0.46),
            drop_wi = Point3f[Point3f(0, 0, 0), Point3f(wi[1], wi[2], 0), Point3f(wi)],
            drop_wo = Point3f[Point3f(0, 0, 0), Point3f(wo[1], wo[2], 0), Point3f(wo)],
            lab_wi = Point3f(1.10 * Vec3d(wi)),
            lab_wo = Point3f(1.10 * Vec3d(wo)),
            lab_ti = Point3f(0.60 * _slerp(n, wi, 0.55)),
            lab_to = Point3f(0.76 * _slerp(n, wo, 0.55)),
            lab_pi = Point3f(0.25 * cos(φi / 2), 0.25 * sin(φi / 2), 0.0),
            lab_po = Point3f(0.56 * cos(φo / 2), 0.56 * sin(φo / 2), 0.0),
            vis_ti = θi > 0.02, vis_to = θo > 0.02,
            vis_pi = φi > 0.05, vis_po = φo > 0.05,
        )
    end
    vis(k::Symbol) = lift(g -> getfield(g, k), geo)

    # -- Guides (below the arrows in draw order) ----------------------------
    lines!(ax, lift(g -> g.drop_wi, geo);
           color = (_RUS_COL.wi, 0.45), linewidth = 1, linestyle = :dash)
    lines!(ax, lift(g -> g.drop_wo, geo);
           color = (_RUS_COL.wo, 0.45), linewidth = 1, linestyle = :dash)
    lines!(ax, lift(g -> g.pi_arc, geo);
           color = _RUS_COL.wi, linewidth = 2, visible = vis(:vis_pi))
    lines!(ax, lift(g -> g.po_arc, geo);
           color = _RUS_COL.wo, linewidth = 2, visible = vis(:vis_po))
    lines!(ax, lift(g -> g.ti_arc, geo);
           color = _RUS_COL.wi, linewidth = 2, visible = vis(:vis_ti))
    lines!(ax, lift(g -> g.to_arc, geo);
           color = _RUS_COL.wo, linewidth = 2, visible = vis(:vis_to))

    # -- Arrows -------------------------------------------------------------
    origin = [Point3f(0, 0, 0)]
    arrows3d!(ax, origin, lift(g -> [g.wi], geo); color = _RUS_COL.wi, _RUS_ARROW...)
    arrows3d!(ax, origin, lift(g -> [g.wo], geo); color = _RUS_COL.wo, _RUS_ARROW...)

    # -- Labels -------------------------------------------------------------
    text!(ax, lift(g -> g.lab_wi, geo); text = L"\omega_i", color = _RUS_COL.wi,
          fontsize = 18, align = (:center, :center))
    text!(ax, lift(g -> g.lab_wo, geo); text = L"\omega_o", color = _RUS_COL.wo,
          fontsize = 18, align = (:center, :center))
    text!(ax, lift(g -> g.lab_ti, geo); text = L"\theta_i", color = _RUS_COL.wi,
          fontsize = 16, align = (:center, :center), visible = vis(:vis_ti))
    text!(ax, lift(g -> g.lab_to, geo); text = L"\theta_o", color = _RUS_COL.wo,
          fontsize = 16, align = (:center, :center), visible = vis(:vis_to))
    text!(ax, lift(g -> g.lab_pi, geo); text = L"\phi_i", color = _RUS_COL.wi,
          fontsize = 16, align = (:center, :center), visible = vis(:vis_pi))
    text!(ax, lift(g -> g.lab_po, geo); text = L"\phi_o", color = _RUS_COL.wo,
          fontsize = 16, align = (:center, :center), visible = vis(:vis_po))

    return ax
end

"""
    draw_rusinkiewicz_construction!(ax, state)

Draw the half/difference construction into `ax`. `state` is a NamedTuple
`(th, ph, td, pd)` in **radians**, or an `Observable` of one (reactive mode).

Elements: the shared stage, the half-vector h, the ωᵢ/ωₒ arrows, the dashed
mirror of ωᵢ, the θh/θd angle arcs, the constant-θd cone around h that φd
travels along (dashed) with the φd arc highlighted on it, the φh arc in the
ground plane, and drop lines to the floor.
"""
function draw_rusinkiewicz_construction!(ax, state)
    state isa Observable || (state = Observable(state))
    _draw_stage!(ax)

    geo = lift(state) do s
        wi, wo, h = rusinkiewicz_to_directions(s.th, s.ph, s.td, s.pd)
        n = Vec3d(0, 0, 1)
        cone = Point3f[Point3f(_rot_hframe(s.th, s.ph, _sph_dir(s.td, φ)))
                       for φ in range(0, 2π; length = 97)]
        pd_arc = s.pd < 0.02 ? Point3f[] :
                 Point3f[Point3f(_rot_hframe(s.th, s.ph, _sph_dir(s.td, φ)))
                         for φ in range(0, s.pd; length = 49)]
        ph_arc = s.ph < 0.02 ? Point3f[] : _ground_arc(s.ph, 0.30)
        (
            wi = Vec3f(wi), wo = Vec3f(wo), h = Vec3f(h),
            th_arc = _arc_points(n, h, 0.45),
            td_arc = _arc_points(h, wi, 0.30),
            cone = cone, pd_arc = pd_arc, ph_arc = ph_arc,
            mirror = Point3f[Point3f(0, 0, 0), Point3f(-wi[1], -wi[2], wi[3])],
            drop_wi = Point3f[Point3f(wi), Point3f(wi[1], wi[2], 0)],
            drop_wo = Point3f[Point3f(wo), Point3f(wo[1], wo[2], 0)],
            lab_wi = Point3f(1.10 * Vec3d(wi)),
            lab_wo = Point3f(1.10 * Vec3d(wo)),
            lab_h  = Point3f(1.13 * h + Vec3d(0.07, 0, 0)),
            lab_mirror = Point3f(0.74 * Vec3d(-wi[1], -wi[2], wi[3])),
            lab_th = Point3f(0.58 * _slerp(n, h, 0.5)),
            lab_td = Point3f(0.44 * _slerp(h, wi, 0.5)),
            lab_pd = Point3f(1.16 * _rot_hframe(s.th, s.ph, _sph_dir(s.td, s.pd / 2))),
            lab_ph = Point3f(0.40 * cos(s.ph / 2), 0.40 * sin(s.ph / 2), 0.0),
        )
    end
    vis_th = lift(s -> s.th > 0.02, state)
    vis_td = lift(s -> s.td > 0.02, state)
    vis_pd = lift(s -> s.pd > 0.05, state)
    vis_ph = lift(s -> s.ph > 0.05, state)

    # -- Guides (below the arrows in draw order) ----------------------------
    lines!(ax, lift(g -> g.cone, geo);
           color = (_RUS_COL.wi, 0.35), linewidth = 1.2, linestyle = :dash)
    lines!(ax, lift(g -> g.pd_arc, geo);
           color = _RUS_COL.wi, linewidth = 2.5, visible = vis_pd)
    lines!(ax, lift(g -> g.ph_arc, geo);
           color = _RUS_COL.h, linewidth = 2, visible = vis_ph)
    lines!(ax, lift(g -> g.mirror, geo);
           color = (_RUS_COL.guide, 0.8), linewidth = 1.2, linestyle = :dash)
    lines!(ax, lift(g -> g.drop_wi, geo);
           color = (_RUS_COL.wi, 0.4), linewidth = 0.8, linestyle = :dot)
    lines!(ax, lift(g -> g.drop_wo, geo);
           color = (_RUS_COL.wo, 0.4), linewidth = 0.8, linestyle = :dot)
    lines!(ax, lift(g -> g.th_arc, geo);
           color = _RUS_COL.h, linewidth = 2, visible = vis_th)
    lines!(ax, lift(g -> g.td_arc, geo);
           color = _RUS_COL.wi, linewidth = 2, visible = vis_td)

    # -- Arrows -------------------------------------------------------------
    origin = [Point3f(0, 0, 0)]
    arrows3d!(ax, origin, lift(g -> [g.h], geo); color = _RUS_COL.h, _RUS_ARROW...)
    arrows3d!(ax, origin, lift(g -> [g.wi], geo); color = _RUS_COL.wi, _RUS_ARROW...)
    arrows3d!(ax, origin, lift(g -> [g.wo], geo); color = _RUS_COL.wo, _RUS_ARROW...)

    # -- Labels -------------------------------------------------------------
    text!(ax, lift(g -> g.lab_h, geo); text = L"h", color = _RUS_COL.h,
          fontsize = 18, align = (:center, :center))
    text!(ax, lift(g -> g.lab_wi, geo); text = L"\omega_i", color = _RUS_COL.wi,
          fontsize = 18, align = (:center, :center))
    text!(ax, lift(g -> g.lab_wo, geo); text = L"\omega_o", color = _RUS_COL.wo,
          fontsize = 18, align = (:center, :center))
    text!(ax, lift(g -> g.lab_mirror, geo); text = "mirror", color = _RUS_COL.guide,
          fontsize = 11, align = (:center, :center))
    text!(ax, lift(g -> g.lab_th, geo); text = L"\theta_h", color = _RUS_COL.h,
          fontsize = 16, align = (:center, :center), visible = vis_th)
    text!(ax, lift(g -> g.lab_td, geo); text = L"\theta_d", color = _RUS_COL.wi,
          fontsize = 16, align = (:center, :center), visible = vis_td)
    text!(ax, lift(g -> g.lab_pd, geo); text = L"\phi_d", color = _RUS_COL.wi,
          fontsize = 16, align = (:center, :center), visible = vis_pd)
    text!(ax, lift(g -> g.lab_ph, geo); text = L"\phi_h", color = _RUS_COL.h,
          fontsize = 16, align = (:center, :center), visible = vis_ph)

    return ax
end

# ============================================================================
# Static figure + export (thesis)
# ============================================================================

_deg(x) = round(rad2deg(x); digits = 1)

# Camera settings are given either once for both panels or as a (left, right) pair.
_cam_pair(x) = x isa Tuple ? (x[1], x[2]) : (x, x)

"""
    rusinkiewicz_figure(θh, φh, θd, φd; azimuth, elevation, size) -> Figure

Two panels of one and the same direction pair: the left in the standard
spherical parameterization (θᵢ, φᵢ, θₒ, φₒ), the right in Rusinkiewicz
coordinates (θh, φh, θd, φd). `azimuth` and `elevation` take a single value
used for both panels, or a `(left, right)` tuple.

The figure carries the two renders and nothing else - no caption, no text
outside the scenes - so it can go into a document as it is.
"""
function rusinkiewicz_figure(θh::Real, φh::Real, θd::Real, φd::Real;
                             azimuth = 0.55, elevation = π / 9,
                             size = (1080, 340))
    az = _cam_pair(azimuth)
    el = _cam_pair(elevation)
    s = (th = Float64(θh), ph = Float64(φh), td = Float64(θd), pd = Float64(φd))

    fig = Figure(; size = size, figure_padding = 2)
    ax_sph = _construction_axis!(fig[1, 1]; azimuth = az[1], elevation = el[1])
    draw_spherical_construction!(ax_sph, s)
    ax_rus = _construction_axis!(fig[1, 2]; azimuth = az[2], elevation = el[2])
    draw_rusinkiewicz_construction!(ax_rus, s)
    colgap!(fig.layout, 0)
    return fig
end

const _RUS_EXPORT_DIR = joinpath(dirname(dirname(@__DIR__)), "exports", "rusinkiewicz")

# Formats offered by the download control, vector first. PDF/SVG keep the panels
# as scalable geometry; PNG/JPG are raster at `px_per_unit` density.
const _RUS_FORMATS = (:pdf, :svg, :png, :jpg)

"""
    _save_figure(path, fig; px_per_unit=4) -> path

Write `fig` to `path`, picking the writer by extension. CairoMakie's `save`
handles PDF, SVG and PNG. It has no JPEG writer (`save` fails with "Unsupported
mime: image/jpeg"), so JPEG goes the long way: render a CairoMakie colorbuffer
(`ARGB32`), drop the alpha to `RGB`, and encode with JpegTurbo.
"""
function _save_figure(path::String, fig; px_per_unit::Real = 4)
    ext = lowercase(splitext(path)[2])
    if ext in (".jpg", ".jpeg")
        img = CairoMakie.Makie.colorbuffer(fig.scene; backend = CairoMakie,
                                           px_per_unit = px_per_unit)
        JpegTurbo.jpeg_encode(path, RGB.(img); quality = 95)
    else
        save(path, fig; backend = CairoMakie, px_per_unit = px_per_unit)
    end
    return path
end

"""
    export_construction(θh, φh, θd, φd;
                        dir=_RUS_EXPORT_DIR, formats=(:pdf,),
                        kwargs...) -> Vector{String}

Render both panels with CairoMakie and save one file per format (see
`_RUS_FORMATS`; raster formats at 4x pixel density). Remaining kwargs are
forwarded to `rusinkiewicz_figure` (azimuth/elevation/size). Returns saved paths.
"""
function export_construction(θh::Real, φh::Real, θd::Real, φd::Real;
                             dir::String = _RUS_EXPORT_DIR,
                             formats = (:pdf,), kwargs...)
    mkpath(dir)
    fig = rusinkiewicz_figure(θh, φh, θd, φd; kwargs...)
    d(x) = round(Int, rad2deg(x))
    base = "rusinkiewicz_th$(d(θh))_ph$(d(φh))_td$(d(θd))_pd$(d(φd))"
    paths = String[]
    for fmt in formats
        p = joinpath(dir, base * "." * String(fmt))
        _save_figure(p, fig)
        push!(paths, p)
    end
    return paths
end

# ============================================================================
# Bonito UI block
# ============================================================================
function _angle_row(label_dom, obs::Observable, maxdeg::Real)
    inp = DOM.input(
        type = "range", min = "0", max = string(maxdeg), step = "0.1",
        value = obs,
        oninput = js"""(event) => $(obs).notify(parseFloat(event.srcElement.value));""",
        style = "flex: 1; min-width: 140px;",
    )
    return DOM.div(
        DOM.span(label_dom; style = "font-size: 0.85em; font-weight: bold; min-width: 30px;"),
        inp,
        DOM.span(map(v -> "$(round(v, digits = 1))°", obs);
                 style = "min-width: 50px; text-align: right; font-size: 0.85em; color: #3498db; font-weight: bold;"),
        style = "display: flex; align-items: center; gap: 8px;",
    )
end

function _slider_group(title::String, accent::String, rows...)
    DOM.div(
        DOM.div(title; style = "font-weight: bold; font-size: 0.85em; color: #34495e; margin-bottom: 4px;"),
        rows...,
        style = "padding: 8px 10px; background: #f8f9fa; border-left: 3px solid $(accent); border-radius: 3px; display: flex; flex-direction: column; gap: 4px; flex: 1; min-width: 220px;",
    )
end

_legend_chip(label, color; dashed::Bool = false) = DOM.span(
    DOM.span(style = "display: inline-block; width: 14px; height: 0; border-top: 3px $(dashed ? "dashed" : "solid") $(color); margin-right: 4px; vertical-align: middle;"),
    DOM.span(label; style = "font-size: 0.8em; color: #555; margin-right: 12px;"),
    style = "display: inline-flex; align-items: center;",
)

"""
    rusinkiewicz_visualization_block(session) -> DOM
"""
function rusinkiewicz_visualization_block(session)
    th0, ph0, td0, pd0 = 20.0, 0.0, 35.0, 90.0
    wi0, wo0, _ = rusinkiewicz_to_directions(deg2rad(th0), deg2rad(ph0),
                                             deg2rad(td0), deg2rad(pd0))
    ti0 = min(_deg(acos(clamp(wi0[3], -1.0, 1.0))), 89.9)
    pi0 = _deg(mod2pi(atan(wi0[2], wi0[1])))
    to0 = min(_deg(acos(clamp(wo0[3], -1.0, 1.0))), 89.9)
    po0 = _deg(mod2pi(atan(wo0[2], wo0[1])))

    o_th = Observable(th0); o_ph = Observable(ph0)
    o_td = Observable(td0); o_pd = Observable(pd0)
    o_ti = Observable(ti0); o_pi = Observable(pi0)
    o_to = Observable(to0); o_po = Observable(po0)

    # θ sliders span 0–89.9° (90° grazing excluded), φ sliders 0–360°.
    row_th = _angle_row(DOM.span("θ", DOM.sub("h")), o_th, 89.9)
    row_ph = _angle_row(DOM.span("φ", DOM.sub("h")), o_ph, 360.0)
    row_td = _angle_row(DOM.span("θ", DOM.sub("d")), o_td, 89.9)
    row_pd = _angle_row(DOM.span("φ", DOM.sub("d")), o_pd, 360.0)
    row_ti = _angle_row(DOM.span("θ", DOM.sub("i")), o_ti, 89.9)
    row_pi = _angle_row(DOM.span("φ", DOM.sub("i")), o_pi, 360.0)
    row_to = _angle_row(DOM.span("θ", DOM.sub("o")), o_to, 89.9)
    row_po = _angle_row(DOM.span("φ", DOM.sub("o")), o_po, 360.0)

    rus_state = Observable((th = deg2rad(th0), ph = deg2rad(ph0),
                            td = deg2rad(td0), pd = deg2rad(pd0)))
    horizon_warn = Observable("")
    syncing = Ref(false)

    function _from_rus()
        s = (th = deg2rad(o_th[]), ph = deg2rad(o_ph[]),
             td = deg2rad(o_td[]), pd = deg2rad(o_pd[]))
        rus_state[] = s
        wi, wo, _ = rusinkiewicz_to_directions(s.th, s.ph, s.td, s.pd)
        below = wo[3] < -1e-9 || wi[3] < -1e-9
        horizon_warn[] = below ?
            "ωₒ is below the horizon - this (θh, θd, φd) tuple is not a valid reflection pair" : ""
        syncing[] = true
        try
            o_ti[] = min(_deg(acos(clamp(wi[3], -1.0, 1.0))), 89.9)
            o_pi[] = _deg(mod2pi(atan(wi[2], wi[1])))
            o_to[] = min(_deg(acos(clamp(wo[3], -1.0, 1.0))), 89.9)
            o_po[] = _deg(mod2pi(atan(wo[2], wo[1])))
        finally
            syncing[] = false
        end
    end

    function _from_sph()
        wi = _sph_dir(deg2rad(o_ti[]), deg2rad(o_pi[]))
        wo = _sph_dir(deg2rad(o_to[]), deg2rad(o_po[]))
        th, ph, td, pd = directions_to_rusinkiewicz(wi, wo)
        rus_state[] = (th = th, ph = ph, td = td, pd = pd)
        horizon_warn[] = ""
        syncing[] = true
        try
            o_th[] = _deg(th)
            o_ph[] = _deg(ph)
            o_td[] = _deg(td)
            o_pd[] = _deg(pd)
        finally
            syncing[] = false
        end
    end

    for o in (o_th, o_ph, o_td, o_pd)
        on(_ -> syncing[] || _from_rus(), o)
    end
    for o in (o_ti, o_pi, o_to, o_po)
        on(_ -> syncing[] || _from_sph(), o)
    end

    # ── Construction figures (spherical | Rusinkiewicz, one shared state) ──
    fig = Figure(size = (880, 300), figure_padding = 2)
    ax_sph = _construction_axis!(fig[1, 1])
    draw_spherical_construction!(ax_sph, rus_state)
    ax_rus = _construction_axis!(fig[1, 2])
    draw_rusinkiewicz_construction!(ax_rus, rus_state)
    colgap!(fig.layout, 0)

    # Dragging either panel rotates both in lockstep.
    cam_syncing = Ref(false)
    for (src, dst) in ((ax_sph, ax_rus), (ax_rus, ax_sph))
        on(src.azimuth) do az
            cam_syncing[] && return
            cam_syncing[] = true
            dst.azimuth[] = az
            cam_syncing[] = false
        end
        on(src.elevation) do el
            cam_syncing[] && return
            cam_syncing[] = true
            dst.elevation[] = el
            cam_syncing[] = false
        end
    end

    # ── Export ─────────────────────────────────────────────────────────────
    export_trigger = Observable(0)
    export_result = Observable{Any}(nothing)
    export_button = DOM.button(
        "Download image",
        onclick = js"""function(event) { $(export_trigger).notify(Date.now()); }""",
        style = "padding: 6px 14px; background: #e67e22; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85em; font-weight: bold; width: fit-content;",
    )
    # Vector by default - the figure is line art, so PDF stays sharp at any size.
    export_format = Observable("pdf")
    _fmt_label = Dict("pdf" => "PDF (vector)", "svg" => "SVG (vector)",
                      "png" => "PNG", "jpg" => "JPG")
    format_select = DOM.select(
        [DOM.option(_fmt_label[String(f)]; value = String(f),
                    selected = (String(f) == export_format[]))
         for f in _RUS_FORMATS]...,
        onchange = js"""function(event) { $(export_format).notify(event.target.value); }""",
        style = "padding: 4px 6px; border: 1px solid #bdc3c7; border-radius: 3px; font-size: 0.8em; background: white;",
    )

    on(export_trigger) do _
        tmp = mktempdir()
        try
            s = rus_state[]
            paths = export_construction(s.th, s.ph, s.td, s.pd;
                                                     dir = tmp,
                                                     formats = (Symbol(export_format[]),),
                                                     azimuth = (ax_sph.azimuth[],
                                                                ax_rus.azimuth[]),
                                                     elevation = (ax_sph.elevation[],
                                                                  ax_rus.elevation[]))
            mimes = Dict(".pdf" => "application/pdf", ".svg" => "image/svg+xml",
                         ".png" => "image/png",       ".jpg" => "image/jpeg")
            files = [(; name = basename(p),
                        mime = get(mimes, lowercase(splitext(p)[2]), "application/octet-stream"),
                        b64 = base64encode(read(p))) for p in paths]
            export_result[] = (ok = true, files = files)
        catch e
            @error "Rusinkiewicz export failed" exception = (e, catch_backtrace())
            export_result[] = (ok = false, msg = sprint(showerror, e))
        finally
            rm(tmp; recursive = true, force = true)
        end
    end

    # Hand the rendered bytes straight to the browser's download feature.
    Bonito.onjs(session, export_result, js"""(r) => {
        if (!r || !r.ok || !r.files) return;
        r.files.forEach((f) => {
            const a = document.createElement('a');
            a.href = 'data:' + f.mime + ';base64,' + f.b64;
            a.download = f.name;
            document.body.appendChild(a);
            a.click();
            a.remove();
        });
    }""")

    export_status = map(export_result) do r
        isnothing(r) && return DOM.div(style = "display: none;")
        if r.ok
            links = [DOM.a(f.name;
                           href = "data:$(f.mime);base64,$(f.b64)",
                           download = f.name,
                           style = "color: #2980b9; font-weight: bold; margin-right: 10px;")
                     for f in r.files]
            DOM.div(
                DOM.span("Downloaded - save again: ";
                         style = "color: #27ae60; margin-right: 6px;"),
                links...,
                style = "padding: 6px 10px; background: #d5f4e6; border-left: 3px solid #27ae60; border-radius: 3px; font-size: 0.8em;",
            )
        else
            DOM.div("Export failed: $(r.msg)";
                    style = "padding: 6px 10px; background: #fadbd8; border-left: 3px solid #c0392b; border-radius: 3px; font-size: 0.8em; color: #c0392b;")
        end
    end

    warn_dom = map(horizon_warn) do w
        isempty(w) && return DOM.div(style = "display: none;")
        DOM.div(w; style = "padding: 6px 10px; background: #fadbd8; border-left: 3px solid #c0392b; border-radius: 3px; font-size: 0.8em; color: #c0392b;")
    end

    # ── Layout ─────────────────────────────────────────────────────────────
    return DOM.div(
        DOM.h3("Rusinkiewicz Parametrization",
               style = "color: #2c3e50; margin: 12px 0 4px 0; font-size: 1em;"),
        DOM.p("Left: the standard parameterization - ωᵢ and ωₒ given by their polar angles " *
              "θᵢ/θₒ from n and their azimuths φᵢ/φₒ from the x axis. Right: the very same " *
              "pair in Rusinkiewicz coordinates, where the half-vector h = (ωᵢ+ωₒ)/‖ωᵢ+ωₒ‖ " *
              "is located by (θh, φh) and the difference vector (θd, φd) is ωᵢ expressed in " *
              "the frame that rotates h to the zenith. Move either slider group, the other " *
              "follows. Hold θh = 0 and vary θd: ωₒ stays locked on the mirror of ωᵢ, which " *
              "is why specular behavior aligns with a single coordinate axis in this " *
              "parametrization.",
              style = "color: #7f8c8d; font-size: 0.85em; margin: 0 0 8px 0; max-width: 900px;"),
        DOM.div(
            DOM.div(
                # Group order mirrors the panels above: spherical drives the left
                # panel, Rusinkiewicz the right one.
                DOM.div(
                    _slider_group("Spherical (incident / outgoing)", "#3498db",
                                  row_ti, row_pi, row_to, row_po),
                    _slider_group("Rusinkiewicz (half / difference angles)", "#e67e22",
                                  row_th, row_ph, row_td, row_pd),
                    style = "display: flex; gap: 8px; flex-wrap: wrap;",
                ),
                warn_dom,
                export_button,
                DOM.div(
                    DOM.span("Format"; style = "font-size: 0.8em; color: #34495e; font-weight: bold;"),
                    format_select,
                    style = "display: flex; align-items: center; gap: 6px;",
                ),
                export_status,
                style = "display: flex; flex-direction: column; gap: 8px; min-width: 300px; max-width: 700px;",
            ),
            DOM.div(
                DOM.div(fig, style = _card_style),
                DOM.div(
                    _legend_chip("ωᵢ incident", "#3498db"),
                    _legend_chip("ωₒ outgoing", "#27ae60"),
                    _legend_chip("h half-vector", "#e67e22"),
                    _legend_chip("mirror of ωᵢ / ideal reflection", "#8c8f90"; dashed = true),
                    _legend_chip("constant-θd cone (φd track)", "#3498db"; dashed = true),
                    style = "margin-top: 4px;",
                ),
                style = "width: fit-content;",
            ),
            style = "display: flex; gap: 12px; align-items: flex-start; flex-wrap: wrap;",
        ),
    )
end

"""
    create_rusinkiewicz_app(session) -> DOM

Full standalone page for the visualization (block + page chrome). Mounted at
`/rusinkiewicz` by `run_platform`, or served on its own by
`run_rusinkiewicz_app`.
"""
function create_rusinkiewicz_app(session)
    return DOM.div(
        rusinkiewicz_visualization_block(session),
        style = "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; " *
                "padding: 20px 24px; max-width: 1200px; margin: 0 auto;",
    )
end

"""
    run_rusinkiewicz_app(; port=8060, host="0.0.0.0", proxy_url=nothing)

Serve just the Rusinkiewicz visualization as a single-tool app. For the full menu
that links all tools together, use `run_platform` instead.
"""
function run_rusinkiewicz_app(; port = 8060, host = "0.0.0.0", proxy_url = nothing)
    app = App(create_rusinkiewicz_app)
    purl = resolve_proxy_url(proxy_url, host, port)

    println("=" ^ 60)
    println("Rusinkiewicz Parametrization Visualization")
    println("=" ^ 60)
    println("local:  http://$host:$port")
    hint = remote_url_hint(purl, port)
    isempty(hint) || println("remote: $hint")
    println("Ctrl+C to stop")
    println("=" ^ 60)

    return Bonito.Server(app, host, port; proxy_url = purl)
end

export rusinkiewicz_visualization_block, rusinkiewicz_figure, export_construction
export rusinkiewicz_to_directions, directions_to_rusinkiewicz
export create_rusinkiewicz_app, run_rusinkiewicz_app
