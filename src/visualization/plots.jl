"""
BRDF Visualization — Central Plotting Module

Provides all visualization primitives and high-level reactive plot handles
for BRDF analysis. Designed for interactive use (Bonito, Pluto, notebooks)
with a thin static convenience wrapper for testing.

Key types and functions:
- `HemispherePlotHandle`: Encapsulates a reactive multi-channel hemisphere scatter plot
- `SurfacePlotHandle`: Encapsulates a reactive luminance-shaped, RGB-colored surface plot
- `setup_hemisphere_plot!`: Create and wire up a hemisphere scatter in a figure position
- `setup_surface_plot!`: Create and wire up a hemisphere surface in a figure position
- `update_hemisphere_plot!` / `update_surface_plot!`: Push new BRDF data into an existing plot
- `create_polar_slice_plot`: Standalone polar slice visualization

Low-level utilities:
- `hemisphere_coords`: Spherical → Cartesian conversion
- `fibonacci_hemisphere`: Uniform hemisphere sampling
- `precompute_hemisphere_coords`: Batch coordinate pre-computation
- `compute_channel_data!`: Multi-threaded per-channel BRDF evaluation (scatter)
- `compute_surface_mesh_data!`: Multi-threaded Fibonacci BRDF evaluation (surface mesh)
- `delaunay_2d` / `triangulate_fibonacci`: 2D Delaunay triangulation for mesh generation

Dependencies: WGLMakie, Colors (no Bonito dependency — that stays in the app layer)
"""

using WGLMakie
using Colors
import WGLMakie.GeometryBasics as GeometryBasics
# import WGLMakie.GeometryBasics: Point3f, TriangleFace, Mesh

# ============================================================================
# LOW-LEVEL UTILITIES
# ============================================================================

"""
    hemisphere_coords(theta::Real, phi::Real) -> NTuple{3, Float32}

Convert spherical coordinates to Cartesian on the unit hemisphere.

- `theta`: Polar angle from zenith (0 to π/2)
- `phi`: Azimuthal angle (0 to 2π)
"""
# Length of the incident-direction indicator line, chosen far larger than any
# plausible hemisphere/BRDF-lobe extent so it reads as an unbounded ray.
# Excluded from Axis3 autolimits (see `xautolimits=false` at the `lines!` call
# sites) so it never affects the plotted view.
const _WI_LINE_LENGTH = 1000.0f0

"""
    update_wi_indicator!(handle, wi_theta, wi_phi)

Update the incident-direction line on a `HemispherePlotHandle` or
`SurfacePlotHandle`. Independent of any loaded BRDF data — call this whenever
the incident direction sliders move, even before/without data being loaded.
"""
function update_wi_indicator!(handle, wi_theta::Float32, wi_phi::Float32)
    wi = hemisphere_coords(wi_theta, wi_phi)
    handle.wi_line[] = [Point3f(0, 0, 0),
                        Point3f(wi[1] * _WI_LINE_LENGTH, wi[2] * _WI_LINE_LENGTH, wi[3] * _WI_LINE_LENGTH)]
end

function hemisphere_coords(theta::Real, phi::Real)
    x = sin(theta) * cos(phi)
    y = sin(theta) * sin(phi)
    z = cos(theta)
    return (Float32(x), Float32(y), Float32(z))
end

"""
    evaluate_brdf_at_direction(data::MeasuredBRDF, wi, wo) -> RGB{Float32}

Evaluate BRDF via the universal `lookup` interface.
"""
function evaluate_brdf_at_direction(data::MeasuredBRDF, wi::Tuple{T,T,T}, wo::Tuple{T,T,T}) where T
    return brdf_value(data, wi, wo)
end

"""
    fibonacci_hemisphere(n::Integer) -> Vector{Tuple{Float32, Float32}}

Generate `n` uniformly distributed (theta, phi) points on the upper hemisphere
using the Fibonacci lattice method.
"""
function fibonacci_hemisphere(n::Integer)
    n_int = Int(n)
    points = Vector{Tuple{Float32, Float32}}(undef, n_int)
    golden_ratio = (1 + sqrt(5)) / 2

    for i in 0:(n_int-1)
        z = Float32(i / (n_int - 1))
        theta = acos(z)
        phi = Float32(mod(2π * i / golden_ratio, 2π))
        points[i+1] = (theta, phi)
    end

    return points
end

"""
    precompute_hemisphere_coords(fib_points) -> Vector{NTuple{3, Float32}}

Batch-convert Fibonacci lattice points to Cartesian coordinates.
"""
function precompute_hemisphere_coords(fib_points::Vector{Tuple{Float32, Float32}})
    n = length(fib_points)
    wo_coords = Vector{NTuple{3, Float32}}(undef, n)
    for (i, (θ, φ)) in enumerate(fib_points)
        wo_coords[i] = hemisphere_coords(θ, φ)
    end
    return wo_coords
end

"""
    apply_radial_scale(value::Float32, scale_factor::Float32, use_log::Bool) -> Float32

Compute the radial scale for a BRDF value.
Linear: `1 + value * scale_factor`
Logarithmic: `1 + log1p(value) * scale_factor`
"""
@inline function apply_radial_scale(value::Float32, scale_factor::Float32, use_log::Bool)
    if use_log
        return 1.0f0 + log1p(value) * scale_factor
    else
        return 1.0f0 + value * scale_factor
    end
end

# ============================================================================
# 2D DELAUNAY TRIANGULATION (Bowyer-Watson)
# ============================================================================

"""
    _in_circumcircle(ax, ay, bx, by, cx, cy, dx, dy) -> Bool

Test whether point (dx, dy) lies inside the circumcircle of triangle (a, b, c).
Assumes the triangle vertices are in counter-clockwise order.
"""
@inline function _in_circumcircle(ax::Float64, ay::Float64, bx::Float64, by::Float64,
                                   cx::Float64, cy::Float64, dx::Float64, dy::Float64)
    adx = ax - dx; ady = ay - dy
    bdx = bx - dx; bdy = by - dy
    cdx = cx - dx; cdy = cy - dy

    det = adx * (bdy * (cdx*cdx + cdy*cdy) - cdy * (bdx*bdx + bdy*bdy)) -
          ady * (bdx * (cdx*cdx + cdy*cdy) - cdx * (bdx*bdx + bdy*bdy)) +
          (adx*adx + ady*ady) * (bdx * cdy - cdx * bdy)

    return det > 0.0
end

"""
    _orient2d(ax, ay, bx, by, cx, cy) -> Float64

Compute the signed area of triangle (a, b, c).
Positive if CCW, negative if CW, zero if collinear.
"""
@inline function _orient2d(ax::Float64, ay::Float64, bx::Float64, by::Float64,
                            cx::Float64, cy::Float64)
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
end

"""
    delaunay_2d(points::Vector{Tuple{Float64, Float64}}) -> Vector{NTuple{3, Int}}

Compute the Delaunay triangulation of a set of 2D points using the
Bowyer-Watson algorithm. Returns face indices (1-based) into `points`.
"""
function delaunay_2d(points::Vector{Tuple{Float64, Float64}})
    n = length(points)
    n < 3 && return NTuple{3, Int}[]

    # Super-triangle enclosing the unit disk (all hemisphere projections)
    st1 = (-5.0, -5.0)
    st2 = ( 5.0, -5.0)
    st3 = ( 0.0,  8.0)

    # All points: 3 super-triangle verts + input points
    all_pts = Vector{Tuple{Float64, Float64}}(undef, n + 3)
    all_pts[1] = st1
    all_pts[2] = st2
    all_pts[3] = st3
    for i in 1:n
        all_pts[i + 3] = points[i]
    end

    # Start with super-triangle (ensure CCW)
    if _orient2d(st1[1], st1[2], st2[1], st2[2], st3[1], st3[2]) < 0
        triangles = Set{NTuple{3, Int}}([(1, 3, 2)])
    else
        triangles = Set{NTuple{3, Int}}([(1, 2, 3)])
    end

    for i in 4:(n + 3)
        px, py = all_pts[i]

        # Find triangles whose circumcircle contains the new point
        bad = NTuple{3, Int}[]
        for t in triangles
            a = all_pts[t[1]]; b = all_pts[t[2]]; c = all_pts[t[3]]
            # Ensure CCW for circumcircle test
            if _orient2d(a[1], a[2], b[1], b[2], c[1], c[2]) > 0
                if _in_circumcircle(a[1], a[2], b[1], b[2], c[1], c[2], px, py)
                    push!(bad, t)
                end
            else
                # CW triangle: flip test sign
                if _in_circumcircle(a[1], a[2], c[1], c[2], b[1], b[2], px, py)
                    push!(bad, t)
                end
            end
        end

        isempty(bad) && continue

        # Collect boundary edges of the cavity (edges not shared by two bad triangles)
        bad_set = Set(bad)
        boundary = Tuple{Int, Int}[]
        for t in bad
            edges = ((t[1], t[2]), (t[2], t[3]), (t[3], t[1]))
            for (ea, eb) in edges
                shared = false
                for t2 in bad
                    t2 === t && continue
                    edges2 = ((t2[1], t2[2]), (t2[2], t2[3]), (t2[3], t2[1]))
                    for (e2a, e2b) in edges2
                        if (ea == e2b && eb == e2a) || (ea == e2a && eb == e2b)
                            shared = true
                            break
                        end
                    end
                    shared && break
                end
                if !shared
                    push!(boundary, (ea, eb))
                end
            end
        end

        # Remove bad triangles
        for t in bad
            delete!(triangles, t)
        end

        # Form new triangles from boundary edges to new point
        for (ea, eb) in boundary
            push!(triangles, (ea, eb, i))
        end
    end

    # Collect result triangles, excluding any that reference super-triangle vertices (1,2,3)
    result = NTuple{3, Int}[]
    for t in triangles
        if t[1] > 3 && t[2] > 3 && t[3] > 3
            push!(result, (t[1] - 3, t[2] - 3, t[3] - 3))
        end
    end

    return result
end

"""
    triangulate_fibonacci(fib_points::Vector{Tuple{Float32, Float32}}) -> Vector{TriangleFace{Int}}

Triangulate Fibonacci hemisphere points by projecting to a 2D disk (orthographic)
and computing the Delaunay triangulation.
"""
function triangulate_fibonacci(fib_points::Vector{Tuple{Float32, Float32}})
    # Project hemisphere points to 2D (orthographic: x, y)
    points_2d = Vector{Tuple{Float64, Float64}}(undef, length(fib_points))
    for (i, (θ, φ)) in enumerate(fib_points)
        points_2d[i] = (Float64(sin(θ) * cos(φ)), Float64(sin(θ) * sin(φ)))
    end

    tri_indices = delaunay_2d(points_2d)
    return [WGLMakie.GeometryBasics.TriangleFace{Int}(t[1], t[2], t[3]) for t in tri_indices]
    
end

"""
    compute_channel_data!(xs, ys, zs, colors, wo_coords, brdf, wi, channel_idx;
                         scale_factor=3.0f0, log_scale=false)

Multi-threaded BRDF evaluation for a single channel.

`channel_idx`: 1=R, 2=G, 3=B, 4=Luminance.
Positions are scaled outward by the radial scale function for visualization.
"""
function compute_channel_data!(xs::Vector{Float32}, ys::Vector{Float32},
                              zs::Vector{Float32}, colors::Vector{Float32},
                              wo_coords::Vector{NTuple{3, Float32}},
                              brdf::MeasuredBRDF, wi::NTuple{3, Float32},
                              channel_idx::Integer;
                              scale_factor::Float32=3.0f0,
                              log_scale::Bool=false)
    n = length(wo_coords)

    Threads.@threads for i in 1:n
        wo = wo_coords[i]
        brdf_rgb = brdf_value(brdf, wi, wo)

        channel_value = if channel_idx == 1
            brdf_rgb.r
        elseif channel_idx == 2
            brdf_rgb.g
        elseif channel_idx == 3
            brdf_rgb.b
        else
            0.299f0 * brdf_rgb.r + 0.587f0 * brdf_rgb.g + 0.114f0 * brdf_rgb.b
        end

        scale = apply_radial_scale(channel_value, scale_factor, log_scale)
        xs[i] = wo[1] * scale
        ys[i] = wo[2] * scale
        zs[i] = wo[3] * scale
        colors[i] = channel_value
    end
end

# ============================================================================
# HEMISPHERE PLOT HANDLE — REACTIVE-FIRST API
# ============================================================================

"""
Channel identifiers used throughout the plotting system.
"""
const CHANNEL_SPECS = (
    r   = (idx=1, colormap=:reds,   label="R",   color_css="#e74c3c"),
    g   = (idx=2, colormap=:greens, label="G",   color_css="#27ae60"),
    b   = (idx=3, colormap=:blues,  label="B",   color_css="#3498db"),
    lum = (idx=4, colormap=:grays,  label="Lum", color_css="#7f8c8d"),
)

"""
    ChannelData

Observable arrays for a single channel's scatter plot data.
"""
struct ChannelData
    xs::Observable{Vector{Float32}}
    ys::Observable{Vector{Float32}}
    zs::Observable{Vector{Float32}}
    colors::Observable{Vector{Float32}}
    visible::Observable{Bool}
end

function ChannelData(n::Int; visible::Bool=true)
    z = () -> zeros(Float32, n)
    return ChannelData(
        Observable(z()), Observable(z()), Observable(z()), Observable(z()),
        Observable(visible)
    )
end

"""
Resize a ChannelData's arrays to length `n`, zeroing them out.
"""
function resize_channel!(ch::ChannelData, n::Int)
    z = zeros(Float32, n)
    ch.xs[]     = copy(z)
    ch.ys[]     = copy(z)
    ch.zs[]     = copy(z)
    ch.colors[] = copy(z)
end

"""
    HemispherePlotHandle

Encapsulates a reactive multi-channel hemisphere scatter plot.

Fields:
- `ax`: The Axis3 object
- `channels`: NamedTuple of ChannelData (:r, :g, :b, :lum)
- `n_points`: Observable{Int} — number of Fibonacci lattice points
- `fib_points`: Cached Fibonacci lattice (regenerated on n_points change)
- `wo_coords`: Cached Cartesian coordinates
- `log_scale`: Observable{Bool} — whether to use logarithmic radial scaling
"""
struct HemispherePlotHandle
    ax::Axis3
    channels::NamedTuple{(:r, :g, :b, :lum), NTuple{4, ChannelData}}
    n_points::Observable{Int}
    fib_points::Observable{Vector{Tuple{Float32, Float32}}}
    wo_coords::Observable{Vector{NTuple{3, Float32}}}
    log_scale::Observable{Bool}
    wi_line::Observable{Vector{Point3f}}
    show_wi::Observable{Bool}
end

"""
    setup_hemisphere_plot!(fig_position; n_points=800,
                          show_r=true, show_g=true, show_b=true, show_lum=false,
                          markersize=8, log_scale=false) -> HemispherePlotHandle

Create a multi-channel hemisphere scatter plot at `fig_position` (e.g. `fig[1,1]`).

Returns a `HemispherePlotHandle` that can be updated reactively via
`update_hemisphere_plot!` or by mutating its observables directly.

The handle manages:
- Fibonacci lattice generation and caching
- Per-channel scatter objects with correct colormaps
- Automatic resizing when `handle.n_points` changes
- Base hemisphere circle
- Optional logarithmic radial scaling

# Arguments
- `fig_position`: Makie figure layout position (e.g. `fig[1,1]`)
- `n_points`: Initial sample count (50–10000, default 800)
- `show_r/g/b/lum`: Initial channel visibility
- `markersize`: Scatter point size
- `log_scale`: Initial log scaling state
"""
function setup_hemisphere_plot!(fig_position;
                                n_points::Int=800,
                                show_r::Bool=true,
                                show_g::Bool=true,
                                show_b::Bool=true,
                                show_lum::Bool=false,
                                markersize::Real=8,
                                log_scale::Bool=false,
                                show_wi::Bool=true)
    @info "Setting up hemisphere plot" n_points threads=Threads.nthreads()

    # Create axis
    ax = Axis3(fig_position,
               aspect=:data,
               title="Load a BRDF file to visualize",
               xlabel="X", ylabel="Y", zlabel="Z",
               azimuth=π/4, elevation=π/6)

    # Create channel data
    channels = (
        r   = ChannelData(n_points; visible=show_r),
        g   = ChannelData(n_points; visible=show_g),
        b   = ChannelData(n_points; visible=show_b),
        lum = ChannelData(n_points; visible=show_lum),
    )

    # Create scatter plots (once — they persist and update via observables)
    for (key, spec) in pairs(CHANNEL_SPECS)
        ch = channels[key]
        scatter!(ax, ch.xs, ch.ys, ch.zs,
                 color=ch.colors, colormap=spec.colormap,
                 markersize=markersize, visible=ch.visible)
    end

    # Base hemisphere circle
    circle_t = range(0, 2π, length=50)
    lines!(ax, cos.(circle_t), sin.(circle_t), zeros(50),
           color=:gray, linewidth=1.5)

    # Fibonacci lattice observables
    n_obs = Observable(n_points)
    fib_obs = Observable(fibonacci_hemisphere(n_points))
    wo_obs = Observable(precompute_hemisphere_coords(fib_obs[]))
    log_obs = Observable(log_scale)

    # Incident direction line: a plain segment through the origin along wi,
    # extended to a virtually-infinite length. xautolimits/yautolimits/zautolimits
    # are set to false so this long line is excluded from Axis3's automatic view
    # fitting (see getlimits() in Makie's axis3d.jl) — otherwise the axis would
    # zoom out to fit it and shrink the hemisphere to a speck.
    wi_line_obs = Observable([Point3f(0, 0, 0), Point3f(0, 0, _WI_LINE_LENGTH)])
    show_wi_obs = Observable(show_wi)
    lines!(ax, wi_line_obs; color=:orange, linewidth=2.5, visible=show_wi_obs,
           xautolimits=false, yautolimits=false, zautolimits=false)

    # Regenerate lattice + resize channels when n_points changes
    on(n_obs) do n_raw
        n = Int(n_raw)
        @info "Resizing hemisphere plot" n_points=n
        fib_obs[] = fibonacci_hemisphere(n)
        wo_obs[]  = precompute_hemisphere_coords(fib_obs[])
        for (_, ch) in pairs(channels)
            resize_channel!(ch, n)
        end
    end

    handle = HemispherePlotHandle(ax, channels, n_obs, fib_obs, wo_obs, log_obs,
                                  wi_line_obs, show_wi_obs)
    @info "Hemisphere plot setup complete"
    return handle
end

"""
    update_hemisphere_plot!(handle::HemispherePlotHandle, brdf::MeasuredBRDF,
                           wi_theta::Float32, wi_phi::Float32)

Recompute and update all visible channels for new BRDF / incident direction.

Only computes channels whose `visible` observable is `true`.
"""
function update_hemisphere_plot!(handle::HemispherePlotHandle,
                                brdf::MeasuredBRDF,
                                wi_theta::Float32, wi_phi::Float32)
    wi = hemisphere_coords(wi_theta, wi_phi)
    n = handle.n_points[]
    coords = handle.wo_coords[]
    use_log = handle.log_scale[]

    handle.ax.title = "BRDF: $(brdf.name)\nIncident: θ=$(round(rad2deg(wi_theta), digits=1))°, φ=$(round(rad2deg(wi_phi), digits=1))°"
    update_wi_indicator!(handle, wi_theta, wi_phi)

    for (key, spec) in pairs(CHANNEL_SPECS)
        ch = handle.channels[key]
        if !ch.visible[]
            continue
        end

        xs = zeros(Float32, n)
        ys = zeros(Float32, n)
        zs = zeros(Float32, n)
        colors = zeros(Float32, n)

        compute_channel_data!(xs, ys, zs, colors, coords, brdf, wi, spec.idx;
                              log_scale=use_log)

        ch.xs[]     = xs
        ch.ys[]     = ys
        ch.zs[]     = zs
        ch.colors[] = colors
    end
end

"""
    clear_hemisphere_plot!(handle::HemispherePlotHandle)

Reset all channels to zeros and set the default title.
"""
function clear_hemisphere_plot!(handle::HemispherePlotHandle)
    handle.ax.title = "Load a BRDF file to visualize"
    n = handle.n_points[]
    for (_, ch) in pairs(handle.channels)
        resize_channel!(ch, n)
    end
end

# ============================================================================
# SURFACE PLOT — FIBONACCI-SAMPLED, RGB-COLORED HEMISPHERE MESH
# ============================================================================

"""
    compute_surface_mesh_data!(vertices, rgb_colors, wo_coords, brdf, wi;
                               scale_factor=3.0f0, log_scale=false)

Multi-threaded BRDF evaluation at Fibonacci lattice points for surface mesh.

Position is scaled by luminance, color is the actual RGB value at each direction.

# Arguments
- `vertices`: Output vector of Point3f (pre-allocated to n_points)
- `rgb_colors`: Output vector of RGBf colors (pre-allocated to n_points)
- `wo_coords`: Cartesian coordinates of Fibonacci lattice points
- `brdf`: MeasuredBRDF instance
- `wi`: Incident direction as (x, y, z)
- `scale_factor`: Luminance-based radial scaling (default: 3.0)
- `log_scale`: Whether to use logarithmic scaling
"""
function compute_surface_mesh_data!(vertices::Vector{Point3f},
                                    rgb_colors::Vector{RGBf},
                                    wo_coords::Vector{NTuple{3, Float32}},
                                    brdf::MeasuredBRDF, wi::NTuple{3, Float32};
                                    scale_factor::Float32=3.0f0,
                                    log_scale::Bool=false)
    n = length(wo_coords)

    Threads.@threads for i in 1:n
        wo = wo_coords[i]
        brdf_rgb = brdf_value(brdf, wi, wo)

        luminance = 0.299f0 * brdf_rgb.r + 0.587f0 * brdf_rgb.g + 0.114f0 * brdf_rgb.b
        scale = apply_radial_scale(luminance, scale_factor, log_scale)

        vertices[i] = Point3f(wo[1] * scale, wo[2] * scale, wo[3] * scale)

        rgb_colors[i] = RGBf(
            clamp(brdf_rgb.r, 0f0, 1f0),
            clamp(brdf_rgb.g, 0f0, 1f0),
            clamp(brdf_rgb.b, 0f0, 1f0)
        )
    end
end

"""
    SurfacePlotHandle

Encapsulates a reactive Fibonacci-sampled, RGB-colored hemisphere surface mesh.

Fields:
- `ax`: The Axis3 object
- `mesh_vertices`: Observable vertex positions (updated per frame)
- `mesh_colors`: Observable per-vertex RGB colors
- `mesh_faces`: Cached triangulation faces (updated when n_points changes)
- `n_points`: Observable sample count
- `fib_points`: Cached Fibonacci lattice (regenerated on n_points change)
- `wo_coords`: Cached Cartesian coordinates
- `log_scale`: Observable{Bool} — whether to use logarithmic radial scaling
"""
struct SurfacePlotHandle
    ax::Axis3
    mesh_vertices::Observable{Vector{Point3f}}
    mesh_colors::Observable{Vector{RGBf}}
    mesh_faces::Observable{Vector{WGLMakie.GeometryBasics.TriangleFace{Int}}}
    n_points::Observable{Int}
    fib_points::Observable{Vector{Tuple{Float32, Float32}}}
    wo_coords::Observable{Vector{NTuple{3, Float32}}}
    log_scale::Observable{Bool}
    wi_line::Observable{Vector{Point3f}}
    show_wi::Observable{Bool}
end

"""
    setup_surface_plot!(fig_position; n_points=800, log_scale=false) -> SurfacePlotHandle

Create a Fibonacci-sampled, RGB-colored hemisphere surface mesh at `fig_position`.

The surface radius at each point is `1 + luminance * scale_factor`, and the
color is the actual RGB value from the BRDF. Uses Fibonacci lattice sampling
(same as the scatter plot) with Delaunay triangulation for the mesh.

# Arguments
- `fig_position`: Makie figure layout position (e.g. `fig[1,1]`)
- `n_points`: Number of Fibonacci lattice sample points (default: 800)
- `log_scale`: Initial log scaling state

# Returns
`SurfacePlotHandle` — update via `update_surface_plot!`
"""
function setup_surface_plot!(fig_position;
                              n_points::Int=800,
                              log_scale::Bool=false,
                              show_wi::Bool=true)
    @info "Setting up surface plot" n_points

    ax = Axis3(fig_position,
               aspect=:data,
               title="Load a BRDF file to visualize",
               xlabel="X", ylabel="Y", zlabel="Z",
               azimuth=π/4, elevation=π/6)

    # Initial Fibonacci lattice and triangulation
    fib = fibonacci_hemisphere(n_points)
    wo = precompute_hemisphere_coords(fib)
    faces = triangulate_fibonacci(fib)

    @info "Surface mesh triangulated" n_points n_faces=length(faces)

    # Observable data
    verts_obs  = Observable([Point3f(c[1], c[2], c[3]) for c in wo])
    colors_obs = Observable(fill(RGBf(0.5, 0.5, 0.5), n_points))
    faces_obs  = Observable(faces)

    n_obs   = Observable(n_points)
    fib_obs = Observable(fib)
    wo_obs  = Observable(wo)
    log_obs = Observable(log_scale)

    # Create mesh (once — updated via observables)
    # Build a reactive Mesh observable from vertices + faces
    mesh_data = map(verts_obs, faces_obs) do v, f
        GeometryBasics.Mesh(v, f)
    end
    mesh!(ax, mesh_data; color=colors_obs, shading=NoShading)

    # Base hemisphere circle
    circle_t = range(0, 2π, length=100)
    lines!(ax, cos.(circle_t), sin.(circle_t), zeros(100),
           color=:gray, linewidth=1.5)

    # Incident direction line (see setup_hemisphere_plot! for the autolimits note)
    wi_line_obs = Observable([Point3f(0, 0, 0), Point3f(0, 0, _WI_LINE_LENGTH)])
    show_wi_obs = Observable(show_wi)
    lines!(ax, wi_line_obs; color=:orange, linewidth=2.5, visible=show_wi_obs,
           xautolimits=false, yautolimits=false, zautolimits=false)

    # Regenerate lattice + triangulation when n_points changes
    on(n_obs) do n_raw
        n = Int(n_raw)
        @info "Resizing surface mesh" n_points=n
        new_fib = fibonacci_hemisphere(n)
        new_wo  = precompute_hemisphere_coords(new_fib)
        new_faces = triangulate_fibonacci(new_fib)
        @info "Surface mesh re-triangulated" n_points=n n_faces=length(new_faces)

        fib_obs[] = new_fib
        wo_obs[]  = new_wo
        faces_obs[] = new_faces
        verts_obs[]  = [Point3f(c[1], c[2], c[3]) for c in new_wo]
        colors_obs[] = fill(RGBf(0.5, 0.5, 0.5), n)
    end

    handle = SurfacePlotHandle(ax, verts_obs, colors_obs, faces_obs,
                                n_obs, fib_obs, wo_obs, log_obs,
                                wi_line_obs, show_wi_obs)
    @info "Surface plot setup complete"
    return handle
end

"""
    update_surface_plot!(handle::SurfacePlotHandle, brdf::MeasuredBRDF,
                        wi_theta::Float32, wi_phi::Float32;
                        scale_factor::Float32=3.0f0)

Recompute the surface mesh for new BRDF / incident direction.
"""
function update_surface_plot!(handle::SurfacePlotHandle,
                              brdf::MeasuredBRDF,
                              wi_theta::Float32, wi_phi::Float32;
                              scale_factor::Float32=3.0f0)
    wi = hemisphere_coords(wi_theta, wi_phi)
    n = handle.n_points[]
    coords = handle.wo_coords[]
    use_log = handle.log_scale[]

    handle.ax.title = "BRDF Surface: $(brdf.name)\nIncident: θ=$(round(rad2deg(wi_theta), digits=1))°, φ=$(round(rad2deg(wi_phi), digits=1))°"
    update_wi_indicator!(handle, wi_theta, wi_phi)

    new_verts  = Vector{Point3f}(undef, n)
    new_colors = Vector{RGBf}(undef, n)

    compute_surface_mesh_data!(new_verts, new_colors, coords, brdf, wi;
                               scale_factor=scale_factor, log_scale=use_log)

    handle.mesh_vertices[] = new_verts
    handle.mesh_colors[]   = new_colors
end

"""
    clear_surface_plot!(handle::SurfacePlotHandle)

Reset surface mesh to unit hemisphere with neutral gray color.
"""
function clear_surface_plot!(handle::SurfacePlotHandle)
    handle.ax.title = "Load a BRDF file to visualize"
    n = handle.n_points[]
    coords = handle.wo_coords[]
    handle.mesh_vertices[] = [Point3f(c[1], c[2], c[3]) for c in coords]
    handle.mesh_colors[]   = fill(RGBf(0.5, 0.5, 0.5), n)
end

# ============================================================================
# SLICE PLOT — PLANE-OF-INCIDENCE POLAR CROSS-SECTION
# ============================================================================
#
# A 2D polar plot of the BRDF sampled along the *plane of incidence*: the plane
# spanned by the surface normal, the incident direction, and the ideal (mirror)
# reflection direction — all three are coplanar. Outgoing directions in this
# plane are parameterized by a signed elevation β ∈ [-π/2, π/2] from the normal:
#   β < 0 → azimuth = wi_phi        (incident / back-scatter side)
#   β = 0 → the surface normal
#   β > 0 → azimuth = wi_phi + π     (reflection / forward side)
# The polar angular coordinate is θ = π/2 - β, so the slice occupies the upper
# semicircle: the normal points up (θ = π/2), grazing angles lie on the surface
# (θ = 0 and θ = π), and the specular lobe should peak near the ideal reflection
# reference line at θ = π/2 - wi_theta.

"""
    _slice_betas(n) -> Vector{Float32}

Signed elevation grid from `-π/2` to `π/2` (see the section header for the
parameterization).
"""
_slice_betas(n::Integer) = collect(Float32, range(-Float32(π)/2, Float32(π)/2, length=Int(n)))

"""
    compute_slice_data!(rs_r, rs_g, rs_b, rs_lum, betas, brdf, wi, wi_phi; log_scale=false)

Multi-threaded evaluation of the BRDF along the plane of incidence. `betas` is a
grid of signed elevation angles from the normal (see section header). Radii are
the raw BRDF values, or `log1p(value)` when `log_scale` is true.
"""
function compute_slice_data!(rs_r::Vector{Float32}, rs_g::Vector{Float32},
                             rs_b::Vector{Float32}, rs_lum::Vector{Float32},
                             betas::Vector{Float32}, brdf::MeasuredBRDF,
                             wi::NTuple{3, Float32}, wi_phi::Float32;
                             log_scale::Bool=false)
    phi_refl = wi_phi + Float32(π)
    n = length(betas)

    Threads.@threads for i in 1:n
        β = betas[i]
        wo = β >= 0 ? hemisphere_coords(β, phi_refl) : hemisphere_coords(-β, wi_phi)
        v = brdf_value(brdf, wi, wo)
        lum = 0.299f0 * v.r + 0.587f0 * v.g + 0.114f0 * v.b
        if log_scale
            rs_r[i]   = log1p(v.r)
            rs_g[i]   = log1p(v.g)
            rs_b[i]   = log1p(v.b)
            rs_lum[i] = log1p(lum)
        else
            rs_r[i]   = v.r
            rs_g[i]   = v.g
            rs_b[i]   = v.b
            rs_lum[i] = lum
        end
    end
end

"""
    SliceChannel

Observable radius array + visibility flag for one channel of the slice plot.
"""
struct SliceChannel
    rs::Observable{Vector{Float32}}
    visible::Observable{Bool}
end

"""
    SlicePlotHandle

Encapsulates a reactive plane-of-incidence polar cross-section.

Fields:
- `ax`: The PolarAxis object (upper semicircle, θ ∈ [0, π])
- `betas` / `thetas`: signed elevation grid and its polar-θ mapping (`π/2 - β`)
- `channels`: NamedTuple of SliceChannel (:r, :g, :b, :lum)
- `n_samples`: Observable{Int} — angular sample count across the semicircle
- `log_scale`: Observable{Bool} — `log1p` radial scaling
- `inc_theta`/`inc_r`, `refl_theta`/`refl_r`: endpoints of the incident-direction
  and ideal-reflection reference lines (radial spokes)
"""
struct SlicePlotHandle
    ax::PolarAxis
    betas::Observable{Vector{Float32}}
    thetas::Observable{Vector{Float32}}
    channels::NamedTuple{(:r, :g, :b, :lum), NTuple{4, SliceChannel}}
    n_samples::Observable{Int}
    log_scale::Observable{Bool}
    inc_theta::Observable{Vector{Float32}}
    inc_r::Observable{Vector{Float32}}
    refl_theta::Observable{Vector{Float32}}
    refl_r::Observable{Vector{Float32}}
end

"""
    setup_slice_plot!(fig_position; n_samples=181, log_scale=false,
                     show_r=true, show_g=true, show_b=true, show_lum=false) -> SlicePlotHandle

Create a reactive plane-of-incidence polar slice at `fig_position`. Update via
`update_slice_plot!` / `update_slice_wi!` or by mutating its observables.
"""
function setup_slice_plot!(fig_position;
                           n_samples::Int=181,
                           log_scale::Bool=false,
                           show_r::Bool=true,
                           show_g::Bool=true,
                           show_b::Bool=true,
                           show_lum::Bool=false)
    @info "Setting up slice plot" n_samples

    ax = PolarAxis(fig_position;
                   title="Load a BRDF file to visualize",
                   thetalimits=(0.0, Float64(π)),
                   rticklabelsize=8,
                   thetaticklabelsize=8)

    betas  = _slice_betas(n_samples)
    thetas = Float32(π)/2 .- betas

    channels = (
        r   = SliceChannel(Observable(zeros(Float32, n_samples)), Observable(show_r)),
        g   = SliceChannel(Observable(zeros(Float32, n_samples)), Observable(show_g)),
        b   = SliceChannel(Observable(zeros(Float32, n_samples)), Observable(show_b)),
        lum = SliceChannel(Observable(zeros(Float32, n_samples)), Observable(show_lum)),
    )

    betas_obs  = Observable(betas)
    thetas_obs = Observable(thetas)
    n_obs      = Observable(n_samples)
    log_obs    = Observable(log_scale)

    # Reference spokes: incident direction (θ = π/2 + wi_theta) and ideal
    # reflection (θ = π/2 - wi_theta). Endpoints are updated reactively; radius
    # tracks the current lobe extent (see update_slice_plot!).
    inc_theta_obs  = Observable(Float32[Float32(π)/2, Float32(π)/2])
    inc_r_obs      = Observable(Float32[0f0, 0f0])
    refl_theta_obs = Observable(Float32[Float32(π)/2, Float32(π)/2])
    refl_r_obs     = Observable(Float32[0f0, 0f0])

    lines!(ax, refl_theta_obs, refl_r_obs; color="#9b59b6", linewidth=1.5, linestyle=:dash)
    lines!(ax, inc_theta_obs,  inc_r_obs;  color=:orange,   linewidth=1.5, linestyle=:dash)

    # Per-channel curves (colors match CHANNEL_SPECS / the UI checkboxes)
    for (key, spec) in pairs(CHANNEL_SPECS)
        ch = channels[key]
        lines!(ax, thetas_obs, ch.rs;
               color=spec.color_css, linewidth=2.5, visible=ch.visible)
    end

    # Regenerate grid + zero curves when the sample count changes
    on(n_obs) do n_raw
        n = Int(n_raw)
        new_betas = _slice_betas(n)
        betas_obs[]  = new_betas
        thetas_obs[] = Float32(π)/2 .- new_betas
        for (_, ch) in pairs(channels)
            ch.rs[] = zeros(Float32, n)
        end
    end

    handle = SlicePlotHandle(ax, betas_obs, thetas_obs, channels, n_obs, log_obs,
                             inc_theta_obs, inc_r_obs, refl_theta_obs, refl_r_obs)
    @info "Slice plot setup complete"
    return handle
end

"""
    update_slice_wi!(handle::SlicePlotHandle, wi_theta::Float32)

Position the incident / ideal-reflection reference spokes for a new incident
elevation. Independent of loaded data — call it whenever the incident direction
changes so the markers stay visible even before a BRDF is loaded.
"""
function update_slice_wi!(handle::SlicePlotHandle, wi_theta::Float32)
    inc_th  = Float32(π)/2 + wi_theta   # incident side (azimuth wi_phi)
    refl_th = Float32(π)/2 - wi_theta   # ideal reflection side (azimuth wi_phi + π)
    # Preserve the current radial extent if the lobe is already drawn.
    r = length(handle.inc_r[]) >= 2 && handle.inc_r[][2] > 0f0 ? handle.inc_r[][2] : 1f0
    handle.inc_theta[]  = Float32[inc_th, inc_th]
    handle.inc_r[]      = Float32[0f0, r]
    handle.refl_theta[] = Float32[refl_th, refl_th]
    handle.refl_r[]     = Float32[0f0, r]
end

"""
    update_slice_plot!(handle::SlicePlotHandle, brdf::MeasuredBRDF,
                      wi_theta::Float32, wi_phi::Float32)

Recompute all channel curves for the plane of incidence and rescale the
reference spokes to the lobe's radial extent.
"""
function update_slice_plot!(handle::SlicePlotHandle, brdf::MeasuredBRDF,
                            wi_theta::Float32, wi_phi::Float32)
    wi = hemisphere_coords(wi_theta, wi_phi)
    n = handle.n_samples[]
    betas = handle.betas[]
    use_log = handle.log_scale[]

    handle.ax.title = "BRDF Slice: $(brdf.name)\nIncident: θ=$(round(rad2deg(wi_theta), digits=1))°, φ=$(round(rad2deg(wi_phi), digits=1))°"

    rs_r   = zeros(Float32, n)
    rs_g   = zeros(Float32, n)
    rs_b   = zeros(Float32, n)
    rs_lum = zeros(Float32, n)
    compute_slice_data!(rs_r, rs_g, rs_b, rs_lum, betas, brdf, wi, wi_phi;
                        log_scale=use_log)

    handle.channels.r.rs[]   = rs_r
    handle.channels.g.rs[]   = rs_g
    handle.channels.b.rs[]   = rs_b
    handle.channels.lum.rs[] = rs_lum

    rmax = max(maximum(rs_r), maximum(rs_g), maximum(rs_b), maximum(rs_lum))
    rmax = (isfinite(rmax) && rmax > 0f0) ? rmax * 1.05f0 : 1f0
    inc_th  = Float32(π)/2 + wi_theta
    refl_th = Float32(π)/2 - wi_theta
    handle.inc_theta[]  = Float32[inc_th, inc_th]
    handle.inc_r[]      = Float32[0f0, rmax]
    handle.refl_theta[] = Float32[refl_th, refl_th]
    handle.refl_r[]     = Float32[0f0, rmax]
end

"""
    clear_slice_plot!(handle::SlicePlotHandle)

Zero all channel curves and reset the title. The incident / reflection reference
spokes are left in place (managed by `update_slice_wi!`).
"""
function clear_slice_plot!(handle::SlicePlotHandle)
    handle.ax.title = "Load a BRDF file to visualize"
    n = handle.n_samples[]
    for (_, ch) in pairs(handle.channels)
        ch.rs[] = zeros(Float32, n)
    end
end

# ============================================================================
# STATIC CONVENIENCE WRAPPER (for testing / notebooks)
# ============================================================================

"""
    create_multichannel_hemisphere_plot(data, wi_theta, wi_phi; n_points=800, kwargs...)

One-shot static figure. Thin wrapper around the reactive API — useful for
testing and notebooks.
"""
function create_multichannel_hemisphere_plot(data::MeasuredBRDF,
                                            wi_theta::Float32,
                                            wi_phi::Float32;
                                            n_points::Integer=800,
                                            show_r::Bool=true,
                                            show_g::Bool=true,
                                            show_b::Bool=true,
                                            show_lum::Bool=false,
                                            markersize::Real=8,
                                            log_scale::Bool=false)
    fig = Figure(resolution=(800, 800))
    handle = setup_hemisphere_plot!(fig[1, 1];
                                    n_points=Int(n_points),
                                    show_r=show_r, show_g=show_g,
                                    show_b=show_b, show_lum=show_lum,
                                    markersize=markersize,
                                    log_scale=log_scale)
    update_hemisphere_plot!(handle, data, wi_theta, wi_phi)
    return fig
end

function create_simple_multichannel_plot(data::MeasuredBRDF;
                                        wi_theta::Float32=Float32(π/4),
                                        wi_phi::Float32=0.0f0,
                                        n_points::Integer=800)
    return create_multichannel_hemisphere_plot(data, wi_theta, wi_phi; n_points=n_points)
end

"""
    create_surface_hemisphere_plot(data, wi_theta, wi_phi; n_points=800)

One-shot static surface figure. Fibonacci-sampled mesh, luminance shapes the
surface, RGB colors it.
"""
function create_surface_hemisphere_plot(data::MeasuredBRDF,
                                       wi_theta::Float32,
                                       wi_phi::Float32;
                                       n_points::Int=800,
                                       scale_factor::Float32=3.0f0,
                                       log_scale::Bool=false)
    fig = Figure(resolution=(800, 800))
    handle = setup_surface_plot!(fig[1, 1]; n_points=n_points, log_scale=log_scale)
    update_surface_plot!(handle, data, wi_theta, wi_phi; scale_factor=scale_factor)
    return fig
end

# ============================================================================
# POLAR SLICE PLOT (standalone — less commonly used interactively)
# ============================================================================

"""
    create_polar_slice_plot(data, wi_theta, wi_phi; theta_out_values, n_phi_samples=180)

Polar + Cartesian plot of BRDF(φ_out) for multiple θ_out slices at fixed incident direction.
"""
function create_polar_slice_plot(data::MeasuredBRDF, wi_theta::Float32, wi_phi::Float32;
                                 theta_out_values::Vector{Float32}=Float32[π/6, π/4, π/3],
                                 n_phi_samples::Int=180)
    @info "Creating polar slice plot" material=data.name wi_theta=wi_theta wi_phi=wi_phi

    wi = hemisphere_coords(wi_theta, wi_phi)
    phi_out_range = range(0, 2π, length=n_phi_samples)
    phi_out_degrees = rad2deg.(phi_out_range)

    fig = Figure(resolution=(1800, 1000))
    colors_palette = cgrad([:red, :blue, :green, :orange, :purple, :brown, :pink, :gray])

    # Polar axes (left)
    ax_r   = PolarAxis(fig[1, 1], title="Red Channel (Polar)", theta_as_x=true)
    ax_g   = PolarAxis(fig[1, 2], title="Green Channel (Polar)", theta_as_x=true)
    ax_b   = PolarAxis(fig[2, 1], title="Blue Channel (Polar)", theta_as_x=true)
    ax_lum = PolarAxis(fig[2, 2], title="Luminance (Polar)", theta_as_x=true)

    # Cartesian axes (right, interactive)
    ax_r_cart   = Axis(fig[1, 3], title="Red Channel", xlabel="φ_out (°)", ylabel="BRDF")
    ax_g_cart   = Axis(fig[1, 4], title="Green Channel", xlabel="φ_out (°)", ylabel="BRDF")
    ax_b_cart   = Axis(fig[2, 3], title="Blue Channel", xlabel="φ_out (°)", ylabel="BRDF")
    ax_lum_cart = Axis(fig[2, 4], title="Luminance", xlabel="φ_out (°)", ylabel="BRDF")

    for (idx, theta_out) in enumerate(theta_out_values)
        r_values   = Float32[]
        g_values   = Float32[]
        b_values   = Float32[]
        lum_values = Float32[]

        for phi_out in phi_out_range
            wo = hemisphere_coords(theta_out, phi_out)
            brdf_value = evaluate_brdf_at_direction(data, wi, wo)
            push!(r_values, brdf_value.r)
            push!(g_values, brdf_value.g)
            push!(b_values, brdf_value.b)
            push!(lum_values, 0.299f0 * brdf_value.r + 0.587f0 * brdf_value.g + 0.114f0 * brdf_value.b)
        end

        line_color = colors_palette[mod1(idx, length(colors_palette))]
        label = "θ=$(round(rad2deg(theta_out), digits=1))°"

        # Polar
        lines!(ax_r,   phi_out_range, r_values,   color=line_color, linewidth=2)
        lines!(ax_g,   phi_out_range, g_values,   color=line_color, linewidth=2)
        lines!(ax_b,   phi_out_range, b_values,   color=line_color, linewidth=2)
        lines!(ax_lum, phi_out_range, lum_values, color=line_color, linewidth=2)

        # Cartesian with inspector
        for (ax_cart, vals, ch_label) in [
            (ax_r_cart, r_values, "R"), (ax_g_cart, g_values, "G"),
            (ax_b_cart, b_values, "B"), (ax_lum_cart, lum_values, "Lum")
        ]
            scatterlines!(ax_cart, phi_out_degrees, vals,
                         color=line_color, linewidth=2, markersize=3,
                         marker=:circle, label=label,
                         inspector_label = (self, i, p) ->
                         "θ_out=$(round(rad2deg(theta_out), digits=1))°\nφ_out=$(round(phi_out_degrees[i], digits=1))°\n$(ch_label)=$(round(vals[i], digits=4))")
        end
    end

    for ax_cart in [ax_r_cart, ax_g_cart, ax_b_cart, ax_lum_cart]
        axislegend(ax_cart, position=:rt, framevisible=false)
    end

    Label(fig[0, :],
          "BRDF Polar Slices: $(data.name) | Incident: θ=$(round(rad2deg(wi_theta), digits=1))°, φ=$(round(rad2deg(wi_phi), digits=1))°",
          fontsize=14, font=:bold)

    DataInspector(fig; depth=9e16, indicator_color=:red, enabled=true)
    return fig
end

function create_simple_polar_slice_plot(data::MeasuredBRDF;
                                        wi_theta::Float32=Float32(π/4),
                                        wi_phi::Float32=0.0f0)
    theta_values = Float32[deg2rad(15), deg2rad(30), deg2rad(45), deg2rad(60), deg2rad(75)]
    return create_polar_slice_plot(data, wi_theta, wi_phi, theta_out_values=theta_values)
end

# ============================================================================
# EXPORTS
# ============================================================================

export hemisphere_coords, evaluate_brdf_at_direction, update_wi_indicator!
export fibonacci_hemisphere, precompute_hemisphere_coords, compute_channel_data!
export apply_radial_scale, delaunay_2d, triangulate_fibonacci
export CHANNEL_SPECS, ChannelData, HemispherePlotHandle
export setup_hemisphere_plot!, update_hemisphere_plot!, clear_hemisphere_plot!
export compute_surface_mesh_data!, SurfacePlotHandle
export setup_surface_plot!, update_surface_plot!, clear_surface_plot!
export SliceChannel, SlicePlotHandle, compute_slice_data!
export setup_slice_plot!, update_slice_plot!, update_slice_wi!, clear_slice_plot!
export create_multichannel_hemisphere_plot, create_simple_multichannel_plot
export create_surface_hemisphere_plot
export create_polar_slice_plot, create_simple_polar_slice_plot