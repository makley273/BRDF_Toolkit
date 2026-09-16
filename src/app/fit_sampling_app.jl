"""
BRDF Fit-Sampling

A reduced Bonito GUI whose ONLY job is to visualize the direction-sampling
strategies the optimizers use during fitting. These strategies are defined in
`src/optimizers/_fit_common.jl` via `_sample_directions(::Symbol, ::Int)`:

- `:rusinkiewicz`  — half-angle, quadratic θ_h bias toward the specular ridge
- `:uniform_half`  — half-angle, uniform θ_h
- `:mixed`         — 80% uniform + 20% near-specular
- `:uniform`       — independent uniform solid-angle (wi, wo); truly unbiased baseline
- `:uniform_cosine`— independent cosine-weighted (wi, wo); crowds the normal
- `:fibonacci`     — √n×√n Fibonacci-lattice product; complete even (wi, wo) grid

The strategies are pure (wi, wo) generators — no BRDF file is needed. This tool
draws the generated samples so a user can see *where* in direction space each
strategy concentrates its measurements and *why* fits differ between strategies.

Crucially, the strategies emit (wi, wo) *pairs*: every outgoing direction is tied
to a specific incoming direction. To make that dependency explicit, the UI offers
incoming-angle sliders (θᵢ, φᵢ) plus a selection cone. The app generates the full
pair set ONCE via `_sample_directions`, then *filters* it down to the pairs whose
wi falls inside the cone around the slider target and shows the corresponding wo.
The conditional outgoing cloud is therefore a literal slice of the strategy's own
output — no sampling logic is re-implemented here (see the `_sample_directions`
note below); the views just analyze and project what the strategy produced.

Linked views (all conditioned on the incoming-angle sliders):
1. Half-angle space scatter (θ_h vs θ_d), colored by φ_d — full set faded, the
   selected subset highlighted. Reveals the specular bias (rusinkiewicz hugs
   θ_h≈0; uniform_half spreads out).
2. Incident hemisphere (wi): full cloud faded, selected wi highlighted, with the
   slider target direction + selection cone drawn for reference.
3. Outgoing hemisphere (wo): the wo of the *selected* pairs only — i.e. where the
   strategy sends energy for the chosen incoming direction — plus the ideal
   mirror direction for comparison.
4. θ_h distribution histogram for the selected subset.

Location: src/app/fit_sampling_app.jl
Entry point: `run_sampling_app()`
"""

using Bonito
using WGLMakie
using StaticArrays
using LinearAlgebra

# `_sample_directions` (from src/optimizers/_fit_common.jl) is loaded into the
# BRDFPlatform module by `load_core_plugins()` at __init__ time. It is referenced
# only inside `recompute!` below, which runs after initialization — so no include
# is needed here, and adding one would fire prematurely during precompilation
# (before the model types the optimizer specs depend on exist).

# ── Strategy catalog (label + one-line description for the UI) ───────────────
const SAMPLING_STRATEGIES = [
    (:rusinkiewicz, "Rusinkiewicz (half-angle, θ_h² bias)",
        "Half-angle sampling with quadratic θ_h remap — concentrates samples on the specular ridge."),
    (:uniform_half, "Uniform half-angle",
        "Half-angle sampling, uniform in θ_h — still centered on the specular ridge but spreads samples across the whole lobe."),
    (:mixed, "Mixed (80% uniform + 20% specular)",
        "Independent uniform (wi, wo) majority for broad coverage, plus a near-specular minority."),
    (:uniform, "Uniform",
        "Independent uniform solid-angle sampling of wi and wo — every equal-area patch of the hemisphere gets equal density."),
    (:uniform_cosine, "Uniform cosine-weighted",
        "Independent cosine-weighted hemisphere sampling of wi and wo (disjoint Halton dims). For any wi the outgoing cloud spans the whole hemisphere, but density crowds the normal (cosθ weighting)."),
    (:fibonacci, "Fibonacci lattice (√n×√n product)",
        "Deterministic golden-spiral lattice of √n directions, taken as the full wi×wo product: every incoming lattice direction paired with every outgoing one. Tiles the whole joint space evenly (lower discrepancy than Halton). Returns m²≈n pairs."),
]

const _STRATEGY_DESC = Dict(string(s) => d for (s, _, d) in SAMPLING_STRATEGIES)

"""
    _analyze_samples(wis, wos) -> NamedTuple

Derive the half-angle coordinates (θ_h, θ_d, φ_d, in degrees) for each (wi, wo)
pair, plus Point3f arrays for the hemisphere scatters. Mirrors the half-angle
frame construction used by the MERL loader and the half-angle sampler so the
displayed coordinates match what the fitter actually traverses.
"""
function _analyze_samples(wis::Vector{SVector{3, Float32}},
                          wos::Vector{SVector{3, Float32}})
    n = length(wis)
    th = Vector{Float32}(undef, n)   # θ_h (deg)
    td = Vector{Float32}(undef, n)   # θ_d (deg)
    pd = Vector{Float32}(undef, n)   # φ_d (deg, [0,360))
    wi_pts = Vector{Point3f}(undef, n)
    wo_pts = Vector{Point3f}(undef, n)

    N = SVector{3, Float32}(0.0f0, 0.0f0, 1.0f0)
    r2d = Float32(180 / π)
    eps = 1.0f-6

    for k in 1:n
        wi = wis[k]; wo = wos[k]
        wi_pts[k] = Point3f(wi[1], wi[2], wi[3])
        wo_pts[k] = Point3f(wo[1], wo[2], wo[3])

        h = wi + wo
        hl = norm(h)
        if hl < eps
            th[k] = 0f0; td[k] = 0f0; pd[k] = 0f0
            continue
        end
        h = h / hl

        theta_h = acos(clamp(h[3], -1f0, 1f0))
        theta_d = acos(clamp(dot(wi, h), -1f0, 1f0))

        # φ_d: azimuth of wi in the (t, b, h) frame (same construction as merl.jl)
        t_raw = N - dot(N, h) * h
        tl = norm(t_raw)
        local t, b
        if tl > eps
            t = t_raw / tl
            b = cross(h, t)
        else
            t = SVector{3, Float32}(1f0, 0f0, 0f0)
            b = SVector{3, Float32}(0f0, 1f0, 0f0)
        end
        phi_d = atan(dot(wi, b), dot(wi, t))
        phi_d = mod(phi_d, Float32(2π))   # full azimuth → [0, 2π), matches the sampler

        th[k] = theta_h * r2d
        td[k] = theta_d * r2d
        pd[k] = phi_d * r2d
    end

    return (th = th, td = td, pd = pd, wi_pts = wi_pts, wo_pts = wo_pts)
end

# Fixed θ_h histogram bins (degrees, 0–90) so the bar layout is stable across
# strategy/count changes — only the heights update reactively.
const _HIST_NBINS = 45
const _HIST_EDGES = collect(range(0f0, 90f0, length = _HIST_NBINS + 1))
const _HIST_CENTERS = Float32[(_HIST_EDGES[i] + _HIST_EDGES[i+1]) / 2 for i in 1:_HIST_NBINS]
const _HIST_WIDTH = Float32(90 / _HIST_NBINS)

function _theta_h_histogram(th::Vector{Float32})
    counts = zeros(Float32, _HIST_NBINS)
    @inbounds for v in th
        b = clamp(floor(Int, v / _HIST_WIDTH) + 1, 1, _HIST_NBINS)
        counts[b] += 1f0
    end
    return counts
end

"""
    _add_base_hemisphere!(ax)

Draw the unit equator circle on an Axis3 for spatial reference.
"""
function _add_base_hemisphere!(ax)
    t = range(0, 2π, length = 96)
    lines!(ax, cos.(t), sin.(t), zeros(length(t)); color = :gray, linewidth = 1.5)
    return ax
end

"""
    _onb(a) -> (u, v)

Branchless orthonormal basis (Duff et al. 2017) completing a unit vector `a`
to a right-handed frame (u, v, a). Used to trace a circle around `a`.
"""
function _onb(a::SVector{3, Float32})
    s = a[3] >= 0f0 ? 1f0 : -1f0
    b = -1f0 / (s + a[3])
    c = a[1] * a[2] * b
    u = SVector{3, Float32}(1f0 + s * a[1] * a[1] * b, s * c, -s * a[1])
    v = SVector{3, Float32}(c, s + a[2] * a[2] * b, -a[2])
    return u, v
end

"""
    _cone_circle(axis, half_angle; n=96) -> Vector{Point3f}

Trace the small circle of unit directions at `half_angle` (rad) from `axis` —
i.e. the boundary of the selection cone, for drawing on the wi hemisphere.
"""
function _cone_circle(axis::SVector{3, Float32}, half_angle::Float32; n::Int = 96)
    u, v = _onb(axis)
    ca, sa = cos(half_angle), sin(half_angle)
    pts = Vector{Point3f}(undef, n + 1)
    for i in 0:n
        t = Float32(2π * i / n)
        d = ca .* axis .+ (sa * cos(t)) .* u .+ (sa * sin(t)) .* v
        pts[i + 1] = Point3f(d[1], d[2], d[3])
    end
    return pts
end

"""
    create_sampling_app(session) -> DOM

Build the reduced fit-sampling visualization UI.
"""
function create_sampling_app(session)
    # ── State ────────────────────────────────────────────────────────────────
    strategy_obs = Observable(string(SAMPLING_STRATEGIES[1][1]))
    n_obs        = Observable(4000)
    marker_obs   = Observable(6)

    # Incoming-direction conditioning (degrees). The wo cloud is the slice of the
    # strategy's pairs whose wi lies within `tol_obs` of this (θᵢ, φᵢ) target.
    theta_i_obs  = Observable(35.0)
    phi_i_obs    = Observable(0.0)
    tol_obs      = Observable(15.0)

    # ── Full pair pool (regenerated only when strategy / N changes) ───────────
    # Kept as plain Refs; the reactive `reselect!` slices these by incoming angle.
    full_wi = Ref(SVector{3, Float32}[])
    full_wo = Ref(SVector{3, Float32}[])
    full_th = Ref(Float32[])
    full_td = Ref(Float32[])
    full_pd = Ref(Float32[])

    # Faded "all samples" backdrops
    full_wi_obs    = Observable(Point3f[])
    full_wo_obs    = Observable(Point3f[])
    full_param_obs = Observable(Point2f[])

    # Selected subset (conditioned on the incoming direction)
    sel_wi_obs       = Observable(Point3f[])
    sel_wo_obs       = Observable(Point3f[])
    sel_wo_color_obs = Observable(Float32[])   # θ_h of selected pairs → wo color
    sel_param_obs    = Observable(Point2f[])
    sel_pd_obs       = Observable(Float32[])
    hist_obs         = Observable(zeros(Float32, _HIST_NBINS))

    # Reference geometry for the incoming direction
    target_pt_obs   = Observable([Point3f(0, 0, 1)])
    target_stem_obs = Observable([Point3f(0, 0, 0), Point3f(0, 0, 1)])
    mirror_pt_obs   = Observable([Point3f(0, 0, 1)])
    cone_obs        = Observable(Point3f[])

    stats_obs = Observable("")

    # Regenerate the full pair pool by DIRECTLY calling the optimizer's sampler.
    function recompute!()
        strat = Symbol(strategy_obs[])
        n = max(1, Int(n_obs[]))
        wis, wos = _sample_directions(strat, n)   # ← the real strategy, no copy
        a = _analyze_samples(wis, wos)

        full_wi[] = wis; full_wo[] = wos
        full_th[] = a.th; full_td[] = a.td; full_pd[] = a.pd

        full_wi_obs[]    = a.wi_pts
        full_wo_obs[]    = a.wo_pts
        full_param_obs[] = [Point2f(a.th[k], a.td[k]) for k in eachindex(a.th)]
    end

    # Slice the pool to pairs whose wi is inside the cone around the slider target.
    function reselect!()
        θ   = Float32(deg2rad(theta_i_obs[]))
        φ   = Float32(deg2rad(phi_i_obs[]))
        tol = Float32(deg2rad(tol_obs[]))
        target = SVector{3, Float32}(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
        mirror = SVector{3, Float32}(-target[1], -target[2], target[3])  # ideal reflection
        costol = cos(tol)

        wis = full_wi[]; wos = full_wo[]
        th = full_th[];  td = full_td[];  pd = full_pd[]

        idx = Int[]
        @inbounds for k in eachindex(wis)
            if dot(wis[k], target) >= costol
                push!(idx, k)
            end
        end

        sel_wi_obs[]       = [Point3f(wis[k][1], wis[k][2], wis[k][3]) for k in idx]
        sel_wo_obs[]       = [Point3f(wos[k][1], wos[k][2], wos[k][3]) for k in idx]
        sel_wo_color_obs[] = Float32[th[k] for k in idx]
        sel_param_obs[]    = [Point2f(th[k], td[k]) for k in idx]
        sel_pd_obs[]       = Float32[pd[k] for k in idx]
        hist_obs[]         = _theta_h_histogram(Float32[th[k] for k in idx])

        target_pt_obs[]   = [Point3f(target[1], target[2], target[3])]
        target_stem_obs[] = [Point3f(0, 0, 0), Point3f(target[1], target[2], target[3])]
        mirror_pt_obs[]   = [Point3f(mirror[1], mirror[2], mirror[3])]
        cone_obs[]        = _cone_circle(target, tol)

        nsel = length(idx)
        if nsel == 0
            stats_obs[] = "0 of $(length(wis)) pairs within $(round(Int, rad2deg(tol)))° of " *
                          "wᵢ=(θ=$(round(Int, theta_i_obs[]))°, φ=$(round(Int, phi_i_obs[]))°) — " *
                          "widen the cone or raise the sample count."
        else
            selth = Float32[th[k] for k in idx]
            mean_th = sum(selth) / nsel
            near_spec = count(<(10f0), selth) / nsel * 100
            # Mean angular spread of the selected wo around the ideal mirror dir.
            dev = 0f0
            @inbounds for k in idx
                dev += acos(clamp(dot(wos[k], mirror), -1f0, 1f0))
            end
            mean_dev = rad2deg(dev / nsel)
            stats_obs[] = "$nsel of $(length(wis)) pairs selected · mean θ_h=$(round(mean_th, digits=1))° · " *
                          "$(round(near_spec, digits=1))% within 10° of specular · " *
                          "wₒ mean spread from mirror=$(round(mean_dev, digits=1))°"
        end
    end

    on(_ -> (recompute!(); reselect!()), strategy_obs)
    on(_ -> (recompute!(); reselect!()), n_obs)
    on(_ -> reselect!(), theta_i_obs)
    on(_ -> reselect!(), phi_i_obs)
    on(_ -> reselect!(), tol_obs)
    recompute!()
    reselect!()

    # ── Figure 1: half-angle scatter (θ_h vs θ_d) — full faded + selected ─────
    fig_param = Figure(resolution = (560, 520))
    ax_param = Axis(fig_param[1, 1];
                    title = "Half-angle sampling: θ_h vs θ_d (selected colored by φ_d)",
                    xlabel = "θ_h  (deg, 0 = specular)", ylabel = "θ_d  (deg)",
                    limits = (0, 90, 0, 90))
    scatter!(ax_param, full_param_obs; color = (:gray, 0.18), markersize = 3)
    sc_param = scatter!(ax_param, sel_param_obs;
                        color = sel_pd_obs, colormap = :plasma, colorrange = (0, 360),
                        markersize = marker_obs)
    Colorbar(fig_param[1, 2], sc_param; label = "φ_d (deg)")

    # ── Figure 2: incident hemisphere (wi) — full faded + selected + cone ─────
    fig_wi = Figure(resolution = (520, 520))
    ax_wi = Axis3(fig_wi[1, 1];
                  aspect = :data, title = "Incident wᵢ  (orange = selected, red = target)",
                  xlabel = "X", ylabel = "Y", zlabel = "Z",
                  azimuth = π/4, elevation = π/6)
    scatter!(ax_wi, full_wi_obs; color = (:gray, 0.12), markersize = 3)
    scatter!(ax_wi, sel_wi_obs; color = :darkorange, markersize = marker_obs)
    lines!(ax_wi, cone_obs; color = :red, linewidth = 2)
    lines!(ax_wi, target_stem_obs; color = :red, linewidth = 2)
    scatter!(ax_wi, target_pt_obs; color = :red, markersize = 16, marker = :star5)
    _add_base_hemisphere!(ax_wi)

    # ── Figure 3: outgoing hemisphere (wo) — selected pairs only ──────────────
    fig_wo = Figure(resolution = (520, 520))
    ax_wo = Axis3(fig_wo[1, 1];
                  aspect = :data, title = "Outgoing wₒ for selected wᵢ  (★ = ideal mirror)",
                  xlabel = "X", ylabel = "Y", zlabel = "Z",
                  azimuth = π/4, elevation = π/6)
    scatter!(ax_wo, full_wo_obs; color = (:gray, 0.06), markersize = 2)
    sc_wo = scatter!(ax_wo, sel_wo_obs;
                     color = sel_wo_color_obs, colormap = :viridis, colorrange = (0, 90),
                     markersize = marker_obs)
    scatter!(ax_wo, mirror_pt_obs; color = :red, markersize = 16, marker = :star5)
    _add_base_hemisphere!(ax_wo)
    Colorbar(fig_wo[1, 2], sc_wo; label = "θ_h (deg)")

    # Each wₒ belongs to the wᵢ directly left of it, so the pair reads best from
    # one shared viewpoint: dragging or zooming either hemisphere moves both.
    # The control-bar checkbox releases the link when the two need to be examined
    # from different angles. (`_sync_axis3_views!` lives in interactive_app.jl.)
    sync_views_obs = Observable(true)
    _sync_axis3_views!(ax_wi, ax_wo; enabled = sync_views_obs)

    # ── Figure 4: θ_h histogram for the selected subset ───────────────────────
    fig_hist = Figure(resolution = (560, 360))
    ax_hist = Axis(fig_hist[1, 1];
                   title = "θ_h distribution (selected subset)",
                   xlabel = "θ_h  (deg, 0 = specular)", ylabel = "sample count",
                   limits = (0, 90, nothing, nothing))
    barplot!(ax_hist, _HIST_CENTERS, hist_obs;
             width = _HIST_WIDTH, color = _HIST_CENTERS,
             colormap = :viridis, colorrange = (0, 90))

    # ── Controls ──────────────────────────────────────────────────────────────
    select_style = "padding: 5px 8px; border: 1px solid #bdc3c7; border-radius: 4px; font-size: 0.9em; background: white;"
    input_style  = "width: 90px; padding: 5px 6px; border: 1px solid #bdc3c7; border-radius: 4px; font-size: 0.9em;"
    label_style  = "font-weight: bold; font-size: 0.9em; margin-right: 6px;"
    val_style    = "font-family: monospace; font-size: 0.85em; color: #2c3e50; min-width: 42px; display: inline-block;"

    strategy_select = DOM.select(
        [DOM.option(label; value = string(sym),
                    selected = (string(sym) == strategy_obs[]))
         for (sym, label, _) in SAMPLING_STRATEGIES]...,
        onchange = js"""function(event) { $(strategy_obs).notify(event.target.value); }""",
        style = select_style,
    )

    n_input = DOM.input(
        type = "number", value = string(n_obs[]), min = "50", max = "50000", step = "250",
        onchange = js"""function(event) {
            const v = parseInt(event.target.value);
            if (v >= 50 && v <= 50000) $(n_obs).notify(v);
        }""",
        style = input_style,
    )

    marker_input = DOM.input(
        type = "number", value = string(marker_obs[]), min = "1", max = "20", step = "1",
        onchange = js"""function(event) {
            const v = parseInt(event.target.value);
            if (v >= 1 && v <= 20) $(marker_obs).notify(v);
        }""",
        style = input_style,
    )

    sync_checkbox = DOM.input(
        type = "checkbox", checked = sync_views_obs[],
        onchange = js"""function(event) { $(sync_views_obs).notify(event.target.checked); }""",
        style = "margin-right: 4px;",
    )

    # Live sliders for the incoming direction + selection cone.
    slider_style = "width: 170px; vertical-align: middle;"
    _slider(obs; mn, mx, st) = DOM.input(
        type = "range", value = string(obs[]),
        min = string(mn), max = string(mx), step = string(st),
        oninput = js"""function(event) { $(obs).notify(parseFloat(event.target.value)); }""",
        style = slider_style,
    )
    theta_i_slider = _slider(theta_i_obs; mn = 0,   mx = 89,  st = 1)
    phi_i_slider   = _slider(phi_i_obs;   mn = 0,   mx = 360, st = 1)
    tol_slider     = _slider(tol_obs;     mn = 3,   mx = 45,  st = 1)

    theta_val = map(v -> DOM.span("θᵢ=$(round(Int, v))°", style = val_style), theta_i_obs)
    phi_val   = map(v -> DOM.span("φᵢ=$(round(Int, v))°", style = val_style), phi_i_obs)
    tol_val   = map(v -> DOM.span("±$(round(Int, v))°",   style = val_style), tol_obs)

    desc_display = map(strategy_obs) do s
        DOM.div(get(_STRATEGY_DESC, s, "");
                style = "color: #5d6d7e; font-size: 0.85em; font-style: italic; margin-top: 6px;")
    end

    stats_display = map(stats_obs) do s
        DOM.div(s; style = "font-family: monospace; color: #2c3e50; font-size: 0.9em; " *
                            "padding: 8px 10px; background: #eaf2f8; border-left: 3px solid #3498db; border-radius: 3px;")
    end

    # ── Layout ────────────────────────────────────────────────────────────────
    panel_style = "background: white; border: 1px solid #dee2e6; border-radius: 6px; padding: 8px; flex: 1; min-width: 0;"

    return DOM.div(
        DOM.h1("BRDF Fit-Sampling",
               style = "color: #2c3e50; margin-bottom: 2px; font-size: 1.5em;"),
        DOM.p("Visualizes the (wᵢ, wₒ) direction-sampling strategies used by the optimizers during fitting · src/optimizers/_fit_common.jl",
              style = "color: #7f8c8d; font-size: 0.9em; margin-top: 0;"),
        DOM.hr(style = "border: 1px solid #ecf0f1; margin-bottom: 12px;"),

        # Control bar
        DOM.div(
            DOM.div(
                DOM.span("Strategy ", style = label_style), strategy_select,
                DOM.span(" Samples ", style = "$label_style margin-left: 16px;"), n_input,
                DOM.span(" Marker size ", style = "$label_style margin-left: 16px;"), marker_input,
                DOM.label(
                    sync_checkbox,
                    DOM.span("Sync wᵢ/wₒ views"; style = "font-size: 0.9em; font-weight: bold;"),
                    style = "display: inline-flex; align-items: center; cursor: pointer; margin-left: 16px;",
                ),
                style = "display: flex; align-items: center; flex-wrap: wrap; gap: 4px;"
            ),
            # Incoming-direction sliders: each wₒ is tied to its wᵢ, so picking an
            # incoming direction shows where that strategy actually sends energy.
            DOM.div(
                DOM.span("Incoming wᵢ ", style = label_style),
                theta_i_slider, theta_val,
                DOM.span(" ", style = "margin-left: 14px;"), phi_i_slider, phi_val,
                DOM.span(" cone ", style = "$label_style margin-left: 14px;"), tol_slider, tol_val,
                style = "display: flex; align-items: center; flex-wrap: wrap; gap: 6px; margin-top: 10px;"
            ),
            desc_display,
            DOM.div(stats_display, style = "margin-top: 8px;"),
            # Sticky like the main app's control bar (see interactive_app.jl):
            # stays in reach while scrolling through the plots below. The opaque
            # background hides content scrolling underneath; the shadow separates
            # it from the plots once stuck.
            style = "padding: 12px; background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 6px; margin-bottom: 12px; " *
                    "position: sticky; top: 0; z-index: 50; box-shadow: 0 2px 6px rgba(0,0,0,0.12);"
        ),

        # Row 1: wi + wo hemispheres
        DOM.div(
            DOM.div(fig_wi, style = panel_style),
            DOM.div(fig_wo, style = "$panel_style margin-left: 10px;"),
            style = "display: flex; gap: 10px; margin-bottom: 12px;"
        ),

        # Row 2: half-angle scatter + histogram
        DOM.div(
            DOM.div(fig_param, style = panel_style),
            DOM.div(fig_hist,  style = "$panel_style margin-left: 10px;"),
            style = "display: flex; gap: 10px;"
        ),

        style = "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; " *
                "padding: 20px; max-width: 1300px; margin: 0 auto;"
    )
end

"""
    run_sampling_app(; port=8050, host="0.0.0.0", proxy_url=nothing)

Start the reduced fit-sampling visualization web app.

When `host="0.0.0.0"`, `proxy_url` is auto-detected as `http://<LAN-IP>:<port>` so
remote devices can load Bonito's runtime/assets. Pass explicitly to override.
"""
function run_sampling_app(; port = 8050, host = "0.0.0.0", proxy_url = nothing)
    app = App(create_sampling_app)
    purl = resolve_proxy_url(proxy_url, host, port)

    println("=" ^ 60)
    println("BRDF Fit-Sampling")
    println("=" ^ 60)
    println("local:  http://$host:$port")
    hint = remote_url_hint(purl, port)
    isempty(hint) || println("remote: $hint")
    println("Ctrl+C to stop")
    println("=" ^ 60)

    return Bonito.Server(app, host, port; proxy_url=purl)
end

export create_sampling_app, run_sampling_app
