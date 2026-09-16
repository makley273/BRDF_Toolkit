"""
Interactive Bonito App with File Upload and Dual Visualization

Web-based interface for BRDF visualization. All plotting logic is delegated
to plots.jl via `HemispherePlotHandle` and `SurfacePlotHandle` — this file
only handles:
- Bonito UI layout and DOM
- File upload / path entry
- Reactive wiring between UI controls and both plot handles

Location: src/app/interactive_app.jl
"""

using Bonito
using WGLMakie
using Base64
import Bonito.TailwindDashboard as D

# Import BRDFPlatform types and functions
# The parent module should handle these imports; included here for standalone use.
# Use @isdefined (checks current module scope) rather than isdefined(Main, ...)
# so this guard works correctly when included as part of BRDFPlatform.
if !@isdefined(MeasuredBRDF)
    include("../types.jl")
    include("../formats/merl.jl")
    include("../visualization/plots.jl")
end

# Universal 4D-LUT generator (generate_lut) — turns any BRDFData into a folded
# 3D texture for the universal renderer. Guard against double-include.
if !@isdefined(generate_lut )
    include("../rendering/universal_texture.jl")
end

# Error-map abstraction (ErrorMap, error_to_measured, AVAILABLE_ERROR_MAPS) and the
# modular visualization blocks. Guarded for standalone include; normally provided
# by BRDFPlatform.jl before this file.
if !@isdefined(ErrorMap)
    include("../errormaps/error_maps.jl")
end
if !@isdefined(render_sphere_block)
    include("viz_blocks.jl")
end

# ── Universal Three.js renderer module ────────────────────────────────────
# Self-contained ES6 module that imports Three.js + OrbitControls from the
# CDN and exports `createUniversalRenderer(container)`. Loaded lazily by Bonito.
const UNIVERSAL_RENDERER = ES6Module(joinpath(@__DIR__, "..", "rendering", "js",
                                          "renderer.js"))

const APP_CSS = Bonito.Asset(joinpath(@__DIR__, "styles", "app.css"))

# ── Universal renderer shader (active path for ALL formats + fitted models) ────
# Read FRESH on every payload build (not cached in a const) so editing
# universal.glsl + reloading a BRDF in the app picks up shader changes WITHOUT a
# Julia restart. (A const froze the shader at module-load time.)
const _UNIVERSAL_GLSL_PATH = joinpath(@__DIR__, "..", "rendering", "shaders", "universal.glsl")
_universal_glsl_code() = try
    read(_UNIVERSAL_GLSL_PATH, String)
catch e
    "// Error loading universal.glsl: $e"
end

# Default 4D-LUT grid = (theta_h, theta_d, phi_d, phi_h). Tunable for
# precision / resolution / anisotropy experiments (theta_h/theta_d/phi_d need
# 8 ≤ N ≤ 128; phi_h may be 1 for isotropic materials; dtype :float16 or :float32).

const UNIVERSAL_LUT_DIMS  = (32,32,32,32)
const UNIVERSAL_LUT_DTYPE = :float16
# Display gain applied in the shader (cosine-weighted). π ≈ render reflectance at
# true albedo; raise for a brighter image, lower if highlights clip too hard.
const UNIVERSAL_LUT_INTENSITY = Float64(π)

# Error-map types (ErrorMap, error_to_measured, AVAILABLE_ERROR_MAPS) now live in
# src/errormaps/error_maps.jl and are included by BRDFPlatform.jl before this file.

const BASE_RESOLUTION   = 400

"""
    _universal_render_payload(brdf; unlit=false) -> Dict

Sample a `BRDFData` (measured or fitted) into the universal 4D LUT and bundle it
with the shader + uniforms for the JS renderer. Returns `nothing` on failure.

`unlit=true` marks the LUT contents as display colors rather than reflectance
(error-map payloads): the shader then skips the cosine/intensity shading and the
Reinhard tone mapping, so the stored colormap colors reach the screen unchanged.
"""
function _universal_render_payload(brdf; unlit::Bool = false,
                                   dims = UNIVERSAL_LUT_DIMS,
                                   dtype::Symbol = UNIVERSAL_LUT_DTYPE)
    try
        b64, cfg = generate_lut(brdf;
                                      dims = dims,
                                      dtype = dtype,
                                      intensity = UNIVERSAL_LUT_INTENSITY)
        return Dict{String,Any}(
            "b64"    => b64,
            "glsl"   => _universal_glsl_code(),
            "config" => cfg,
            "uniforms" => Dict{String,Any}(
                "uGridDims"  => Float64.(cfg["dims"]),
                "uIntensity" => cfg["intensity"],
                "uUnlit"     => unlit ? 1.0 : 0.0,
            ),
        )
    catch e
        @warn "Failed to create LUT" exception=(e, catch_backtrace())
        return nothing
    end
end

# ── Analytical-model → MeasuredBRDF adapter ───────────────────────────────
# The plot handles in plots.jl take a MeasuredBRDF (they call brdf_value
# internally, which dispatches on `brdf._data` via `_brdf_value`). Wrapping
# a fitted BRDFModel behind this tiny adapter lets the same plot handles
# visualize the fitted model without any changes to plots.jl.
struct _AnalyticalWrapper
    model::BRDFModel
end

function _brdf_value(w::_AnalyticalWrapper,
                     ::Vector{Float32},
                     wi::SVector{3, Float32},
                     wo::SVector{3, Float32})
    v = evaluate(w.model, wi, wo)
    return RGB{Float32}(Float32(v.r), Float32(v.g), Float32(v.b))
end

n_samples_str(::_AnalyticalWrapper) = "analytical"

function _model_to_measured(model::BRDFModel,
                            wavelengths::Vector{Float32},
                            source_name::String)::MeasuredBRDF
    return MeasuredBRDF(
        _AnalyticalWrapper(model),
        wavelengths,
        "fit:" * source_name;
        source_format = :fitted,
        status = :ok,
    )
end

"""
    _sync_axis3_views!(axes...)

Keep the camera view of several `Axis3` plots in lockstep. We mirror three
Observables from each axis onto the others:

- `azimuth` / `elevation` — rotation (they update live as the user drags).
- `targetlimits` — in the default `:fitzoom` viewmode, scroll-zoom does **not**
  touch `zoom_mult` (the `:fitzoom` projection re-normalizes the box to the
  viewport, cancelling it); it shrinks/grows `targetlimits` instead. So syncing
  `targetlimits` is what propagates zoom across the plots.

Each property gets its own guard flag so writing a peer's Observable can't echo
back into an update loop. With fewer than two axes this is a no-op.

`enabled` (an `Observable{Bool}`, or `nothing` for "always on") makes the link
switchable: while it reads `false` the handlers pass without propagating, and
flipping it back to `true` snaps the peers onto the first axis so the plots line
up at once instead of only at the next drag.

Caveat: `targetlimits` doubles as the data-fit extent, so this also couples the
plots' framing — when one plot reframes to fit freshly loaded data, the others
follow. For the measured/fitted hemisphere plots (same radial scaling, similar
extents) that shared scale is fine and aids comparison.
"""
function _sync_axis3_views!(axes...; enabled = nothing)
    length(axes) < 2 && return nothing
    _linked() = enabled === nothing || enabled[]
    for prop in (:azimuth, :elevation, :targetlimits)
        syncing = Ref(false)
        for src in axes
            on(getproperty(src, prop)) do v
                (_linked() && !syncing[]) || return
                syncing[] = true
                try
                    for dst in axes
                        dst === src && continue
                        getproperty(dst, prop)[] = v
                    end
                finally
                    syncing[] = false
                end
            end
        end
    end
    if enabled !== nothing
        on(enabled) do linked
            linked || return
            src = first(axes)
            for prop in (:azimuth, :elevation, :targetlimits)
                v = getproperty(src, prop)[]
                for dst in axes
                    dst === src && continue
                    getproperty(dst, prop)[] = v
                end
            end
        end
    end
    return nothing
end

"""
    _sync_polar_views!(axes...)

Keep the view of several `PolarAxis` plots in lockstep — the polar counterpart of
`_sync_axis3_views!`. `PolarAxis` has no camera; its interactions (scroll zoom,
right-drag pan, axis rotation) write the `target_*` Observables directly, so
mirroring those four propagates every view change:

- `target_rlims` / `target_thetalims` — radial and angular zoom extent.
- `target_theta_0` / `target_r0` — axis rotation and radial origin.

Each property gets its own guard flag so writing a peer's Observable can't echo
back into an update loop. With fewer than two axes this is a no-op.

Caveat (same as the Axis3 version): `target_rlims` doubles as the autolimit
extent, so the measured and fitted slices also share a radial scale — when one
reframes to freshly loaded data the other follows. That shared scale is what
makes the two polar plots directly comparable.
"""
function _sync_polar_views!(axes...)
    length(axes) < 2 && return nothing
    for prop in (:target_rlims, :target_thetalims, :target_theta_0, :target_r0)
        syncing = Ref(false)
        for src in axes
            on(getproperty(src, prop)) do v
                syncing[] && return
                syncing[] = true
                try
                    for dst in axes
                        dst === src && continue
                        getproperty(dst, prop)[] = v
                    end
                finally
                    syncing[] = false
                end
            end
        end
    end
    return nothing
end

"""
    create_interactive_app(session) -> DOM

Build the full Bonito application: file upload, controls, scatter + surface plots.

All visualization is managed through `HemispherePlotHandle` and `SurfacePlotHandle`
from plots.jl — this file only handles UI layout and reactive wiring.
"""
function create_interactive_app(session)
    # ── State observables ───────────────────────────────────────────────────
    loaded_brdf = Observable{Union{Nothing, MeasuredBRDF}}(nothing)
    upload_status = Observable{String}("idle")
    wi_theta = Observable(Float32(π/4))
    wi_phi = Observable(Float32(0.0))
    uploaded_file_data = Observable{Union{Nothing, Dict{String, Any}}}(nothing)
    file_path_obs = Observable{Union{Nothing, String}}(nothing)
    
    # ── Universal LUT resolution (interactive) ──────────────────────────────
    # Live 4D-LUT grid resolution + precision. Changing `lut_dims_obs` regenerates
    # all three spheres from their source BRDFs. The `build_payload` closure captures
    # the current resolution so every payload stays in lock-step with the UI.
    # NOTE the folded texture depth = N_phi_d × N_phi_h must stay ≤ 2048 (WebGL2
    # MAX_3D_TEXTURE_SIZE)
    lut_dims_obs  = Observable{NTuple{4,Int}}(UNIVERSAL_LUT_DIMS)
    lut_dtype_obs = Observable{Symbol}(UNIVERSAL_LUT_DTYPE)
    build_payload(brdf; unlit::Bool = false) =
        _universal_render_payload(brdf; unlit = unlit,
                                  dims = lut_dims_obs[], dtype = lut_dtype_obs[])

    # "Pending" edits from the resolution inputs. Applied atomically by the Apply
    # button (one notify → one rebuild), so mid-edit values never trigger a rebuild.
    res_th = Observable(UNIVERSAL_LUT_DIMS[1])   # N_theta_h
    res_td = Observable(UNIVERSAL_LUT_DIMS[2])   # N_theta_d
    res_pd = Observable(UNIVERSAL_LUT_DIMS[3])   # N_phi_d
    res_ph = Observable(UNIVERSAL_LUT_DIMS[4])   # N_phi_h (anisotropy)
    res_dtype_pending = Observable{String}(String(UNIVERSAL_LUT_DTYPE))  # JS <select> sends a string
    res_apply_trigger = Observable(0)
    res_status = Observable{String}("")          # Apply feedback line

    _MAX_3D_TEX = 2048                           # WebGL2 MAX_3D_TEXTURE_SIZE floor
    on(res_apply_trigger) do _
        # Clamp per-axis to the LUT generator's asserts (8–128; phi_h may be 1).
        a = clamp(res_th[], 8, 128)
        b = clamp(res_td[], 8, 128)
        c = clamp(res_pd[], 8, 128)
        d = clamp(res_ph[], 1, 128)
        depth = c * d
        if depth > _MAX_3D_TEX
            res_status[] = "❌ folded depth φd×φh = $depth > $_MAX_3D_TEX (GPU 3D-texture limit). Lower φd or φh."
            return
        end
        dt = Symbol(res_dtype_pending[])
        res_status[] = "✅"
        lut_dtype_obs[] = dt                     # silent (no handler); read by next rebuild
        lut_dims_obs[]  = (a, b, c, d)           # fires _rebuild_all_luts
    end

    # Live readout of the *pending* grid: folded depth vs the GPU cap and the
    # GPU-side memory footprint (RGBA; 2 B/channel float16, 4 B/channel float32).
    res_info = map(res_th, res_td, res_pd, res_ph, res_dtype_pending) do a, b, c, d, dt
        depth    = c * d
        texels   = a * b * c * d
        bytes_px = dt == "float16" ? 8 : 16
        mb       = texels * bytes_px / 1_048_576
        over     = depth > 2048
        col      = over ? "#c0392b" : "#7f8c8d"
        DOM.span("depth φd×φh = $depth" * (over ? " ✗>2048" : " ✓") *
                 " · $(round(mb; digits = 1)) MB GPU",
                 style = "font-size: 0.8em; color: $col; margin-left: 8px;")
    end

    # Visibility toggles for visualization panels
    show_scatter = Observable{Bool}(true)
    show_surface = Observable{Bool}(true)
    show_slice = Observable{Bool}(true)
    show_render = Observable{Bool}(true)
    show_wi = Observable{Bool}(true)

    # Error-map selection (drives the third "Error Map" sphere). `selected_errormap`
    # is the canonical ErrorMap; the dropdown notifies the index `selected_errormap_idx`.
    selected_errormap = Observable{ErrorMap}(AVAILABLE_ERROR_MAPS[1])
    selected_errormap_idx = Observable{Int}(1)
    on(selected_errormap_idx) do i
        if 1 <= i <= length(AVAILABLE_ERROR_MAPS)
            selected_errormap[] = AVAILABLE_ERROR_MAPS[i]
        end
    end

    # Shape of the rendered preview object (drives all three Three.js panels).
    # Keys must match GEOMETRY_FACTORIES in renderer.js; values are display labels.
    SHAPE_LABELS = ["sphere" => "Sphere", "cube" => "Cube",
                    "cylinder" => "Cylinder", "cone" => "Cone",
                    "torus" => "Torus", "torusknot" => "Torus Knot",
                    "icosahedron" => "Icosahedron"]
    render_shape = Observable{String}("sphere")
    # Reactive "<Shape> Render" heading text shared by all three render panels.
    shape_title = map(s -> "$(Dict(SHAPE_LABELS)[s]) Render", render_shape)

    # ── File upload handling ────────────────────────────────────────────────
    file_input_element = DOM.input(
        type="file",
        accept=".binary,.bird,.json,.bin,.bsdf",
        id="brdf-file-input",
        style="padding: 6px; border: 1px solid #bdc3c7; border-radius: 4px; cursor: pointer; font-size: 0.85em; width: fit-content;",
        onchange=js"""
            async function(event) {
                const file = event.target.files[0];
                if (!file) return;
                try {
                    $(upload_status).notify('uploading');
                    const arrayBuffer = await file.arrayBuffer();
                    const uint8Array = new Uint8Array(arrayBuffer);
                    let binary = '';
                    const chunkSize = 0x8000;
                    for (let i = 0; i < uint8Array.length; i += chunkSize) {
                        binary += String.fromCharCode.apply(null,
                            uint8Array.subarray(i, Math.min(i + chunkSize, uint8Array.length)));
                    }
                    $(upload_status).notify('processing');
                    $(uploaded_file_data).notify({ filename: file.name, data: btoa(binary) });
                } catch (error) {
                    console.error('Error processing file:', error);
                    $(upload_status).notify('error');
                }
            }
        """
    )

    on(uploaded_file_data) do file_data
        try
            isnothing(file_data) && return
            filename = file_data["filename"]
            file_bytes = base64decode(file_data["data"])
            temp_path = joinpath(mktempdir(), filename)
            write(temp_path, file_bytes)
            data = load_brdf(temp_path)
            if data.status == :ok
                loaded_brdf[] = data
                upload_status[] = "complete"
                # Sample into the universal 4D LUT — works for every format.
                measured_render_payload[] = build_payload(data)
                @info "Loaded uploaded file: $(data.name)"
            else
                @warn "Failed to load BRDF: $(data.error_msg)"
                loaded_brdf[] = nothing
                measured_render_payload[] = nothing
                upload_status[] = "error"
            end
        catch e
            @error "Error processing upload" exception=(e, catch_backtrace())
            loaded_brdf[] = nothing
            measured_render_payload[] = nothing
            upload_status[] = "error"
        end
    end

    # Manual path entry
    # file_path_input = DOM.input(
    #     type="text",
    #     placeholder="/path/to/file.binary",
    #     style="flex: 1; padding: 6px; border: 1px solid #ccc; border-radius: 4px; font-size: 0.85em; width: fit-content;"
    # )
    # load_button = DOM.button(
    #     "Load",
    #     onclick=js"""function(event) {
    #         const path = $(file_path_input).value;
    #         if (path) $(file_path_obs).notify(path);
    #     }""",
    #     style="padding: 6px 14px; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85em; width: fit-content;"
    # )

    on(file_path_obs) do filepath
        try
            (isnothing(filepath) || isempty(filepath)) && return
            !isfile(filepath) && (@warn "File not found: $filepath"; return)
            data = load_brdf(filepath)
            if data.status == :ok
                loaded_brdf[] = data
                # Sample into the universal 4D LUT — works for every format.
                measured_render_payload[] = build_payload(data)
                @info "Loaded: $(data.name)"
            else
                @warn "Failed to load BRDF: $(data.error_msg)"
                loaded_brdf[] = nothing
                measured_render_payload[] = nothing
            end
        catch e
            @error "Error loading file" exception=(e, catch_backtrace())
            loaded_brdf[] = nothing
            measured_render_payload[] = nothing
        end
    end

    # ── Upload status indicator ─────────────────────────────────────────────
    upload_indicator = map(upload_status) do status
        styles = Dict(
            "uploading"  => ("⏳ Uploading...",   "#3498db", "#ebf5fb"),
            "processing" => ("🔄 Processing...",  "#f39c12", "#fef5e7"),
            "error"      => ("❌ Upload error.",   "#e74c3c", "#fadbd8"),
        )
        if haskey(styles, status)
            msg, fg, bg = styles[status]
            DOM.div(msg, style="color: $fg; padding: 6px 10px; background: $bg; border-left: 3px solid $fg; border-radius: 3px; margin-top: 6px; font-size: 0.85em; width: fit-content;")
        else
            DOM.div(style="display: none;")
        end
    end

    # ── Loaded file status (compact) ────────────────────────────────────────
    status_display = map(loaded_brdf) do brdf
        if isnothing(brdf)
            return DOM.div("📁 No file loaded.", style="color: #7f8c8d; padding: 6px; font-style: italic; font-size: 0.85em; width: fit-content;")
        end
        n_samples = n_samples_str(brdf._data)
        DOM.div(
            DOM.span("✅ $(brdf.name)", style="color: #27ae60; font-weight: bold; margin-right: 12px;"),
            DOM.span("$(brdf.source_format)", style="margin-right: 12px;"),
            DOM.span("$(n_samples) samples", style="margin-right: 12px;"),
            DOM.span("λ: $(join(Int.(brdf.wavelengths), ", ")) nm", style="margin-right: 12px;"),
            DOM.span("$(Threads.nthreads()) threads", style="color: #3498db; font-style: italic;"),
            style="padding: 8px 10px; background: #d5f4e6; border-left: 3px solid #27ae60; border-radius: 3px; margin-top: 6px; font-size: 0.85em;"
        )
    end

    # ── Angle sliders ───────────────────────────────────────────────────────
    theta_slider = Bonito.Slider(0:0.01:π/2)
    phi_slider   = Bonito.Slider(0:0.01:2π)
    theta_slider.value[] = Float64(wi_theta[])
    phi_slider.value[]   = Float64(wi_phi[])
    on(theta_slider.value) do val; wi_theta[] = Float32(val); end
    on(phi_slider.value)   do val; wi_phi[]   = Float32(val); end

    # ── Render gain slider (Measured + Fitted spheres; NOT the error map) ───
    # Photographic stops: displayed gain = default intensity (π) × 2^stops, so
    # ±1 slider unit halves/doubles brightness — evaluates dark materials
    # without re-baking (uIntensity is a live shader uniform).
    gain_slider = Bonito.Slider(-3:0.25:3)
    gain_slider.value[] = 0.0
    render_gain_obs = map(gain_slider.value) do s
        UNIVERSAL_LUT_INTENSITY * 2.0^Float64(s)
    end

    # ── Hemisphere scatter + surface plots: measured ───────────────────────
    fig_scatter = Figure(resolution=(BASE_RESOLUTION, BASE_RESOLUTION))
    scatter_handle = setup_hemisphere_plot!(fig_scatter[1, 1];
                                            n_points=800,
                                            show_r=true, show_g=true, show_b=true, show_lum=false)

    fig_surface = Figure(resolution=(BASE_RESOLUTION, BASE_RESOLUTION))
    surface_handle = setup_surface_plot!(fig_surface[1, 1]; n_points=800)

    fig_slice = Figure(resolution=(BASE_RESOLUTION, BASE_RESOLUTION))
    slice_handle = setup_slice_plot!(fig_slice[1, 1];
                                     show_r=true, show_g=true, show_b=true, show_lum=false)

    # ── Hemisphere scatter + surface plots: fitted model ───────────────────
    fig_scatter_fit = Figure(resolution=(BASE_RESOLUTION, BASE_RESOLUTION))
    scatter_handle_fit = setup_hemisphere_plot!(fig_scatter_fit[1, 1];
                                                n_points=800,
                                                show_r=true, show_g=true, show_b=true, show_lum=false)

    fig_surface_fit = Figure(resolution=(BASE_RESOLUTION, BASE_RESOLUTION))
    surface_handle_fit = setup_surface_plot!(fig_surface_fit[1, 1]; n_points=800)

    fig_slice_fit = Figure(resolution=(BASE_RESOLUTION, BASE_RESOLUTION))
    slice_handle_fit = setup_slice_plot!(fig_slice_fit[1, 1];
                                         show_r=true, show_g=true, show_b=true, show_lum=false)

    # ── Three.js renderer state ────────────────────────────────────────────
    # Bundle b64 + glsl + config together so the GLSL string is delivered via
    # msgpack (same as fitted_render_obs), never embedded raw in JS source code.
    # Embedding multi-line GLSL inline via $(...) causes SyntaxError in
    # Bonito's new Function(...) path regardless of Observable wrapping.
    measured_render_payload = Observable{Union{Nothing, Dict{String, Any}}}(nothing)
    fitted_render_obs = Observable{Any}(nothing)
    error_render_obs  = Observable{Any}(nothing)

    # ── Fit state ──────────────────────────────────────────────────────────
    model_names     = sort(collect(keys(MODEL_REGISTRY)))
    optimizer_names = sort(collect(keys(OPTIMIZER_REGISTRY)))
    # Default optimizer: LeastSquaresOptim with 16 restarts (see lm_LSOF.jl). The
    # thesis optimizer sweep has it reach the converged SSIM ~5x faster than the
    # single-start fminbox L-BFGS. Falls back to the first registered if not loaded.
    default_optimizer = :leastsquaresoptimfitter in optimizer_names ? :leastsquaresoptimfitter :
                        (isempty(optimizer_names) ? "" : optimizer_names[1])
    selected_model_name     = Observable(string(isempty(model_names) ? "" : model_names[1]))
    selected_optimizer_name = Observable(string(default_optimizer))
    fitted_model  = Observable{Union{Nothing, BRDFModel}}(nothing)
    fitted_brdf   = Observable{Union{Nothing, MeasuredBRDF}}(nothing)
    fit_status    = Observable{String}("idle")
    fit_rmse      = Observable{Union{Nothing, Float64}}(nothing)
    fit_crwe      = Observable{Union{Nothing, Float64}}(nothing)
    fit_rmsle      = Observable{Union{Nothing, Float64}}(nothing)
    fit_ssim      = Observable{Union{Nothing, NamedTuple}}(nothing)

    # When a new model is fitted, wrap it for the plot handles and trigger the JS render
    on(fitted_model) do m
        if isnothing(m)
            fitted_brdf[] = nothing
            fitted_render_obs[] = nothing
            error_render_obs[] = nothing
        else
            # 1. Update Julia Visualization
            source = isnothing(loaded_brdf[]) ? "unknown" : loaded_brdf[].name
            wls    = isnothing(loaded_brdf[]) ? Float32[630, 532, 465] : loaded_brdf[].wavelengths
            wrapped = _model_to_measured(m, wls, source)
            fitted_brdf[] = wrapped

            # 2. Update JS Render Canvas — sample the fitted model into the same
            #    universal 4D LUT used for measured data (one renderer for all).
            fitted_render_obs[] = build_payload(wrapped)

            # 3. Error-map sphere — relative error of measured vs fitted, sampled
            #    through the same path. Needs the measured data; skip if absent.
            if isnothing(loaded_brdf[])
                error_render_obs[] = nothing
            else
                err_wrapped = error_to_measured(loaded_brdf[], m, selected_errormap[], wls, source)
                error_render_obs[] = build_payload(err_wrapped; unlit = true)
            end
        end
    end

    # Regenerate the error-map sphere when the chosen ErrorMap changes (no re-fit).
    on(selected_errormap) do em
        m = fitted_model[]
        data = loaded_brdf[]
        if !isnothing(m) && !isnothing(data)
            err_wrapped = error_to_measured(data, m, em, data.wavelengths, data.name)
            error_render_obs[] = build_payload(err_wrapped; unlit = true)
        end
    end

    # Regenerate every sphere when the LUT resolution / precision changes. Reads the
    # source objects still held in their observables (measured data, fitted model)
    # so nothing needs to be re-loaded or re-fitted.
    function _rebuild_all_luts()
        d = loaded_brdf[]
        measured_render_payload[] = isnothing(d) ? nothing : build_payload(d)
        fb = fitted_brdf[]
        fitted_render_obs[] = isnothing(fb) ? nothing : build_payload(fb)
        m = fitted_model[]
        if !isnothing(m) && !isnothing(d)
            err_wrapped = error_to_measured(d, m, selected_errormap[], d.wavelengths, d.name)
            error_render_obs[] = build_payload(err_wrapped; unlit = true)
        end
    end
    on(_ -> _rebuild_all_luts(), lut_dims_obs)

    # Invalidate fit when the underlying data changes
    on(loaded_brdf) do _
        fitted_model[] = nothing
        fit_status[] = "idle"
        fit_rmse[] = nothing
        fit_crwe[] = nothing
        fit_rmsle[] = nothing
        fit_ssim[] = nothing
    end

    # ── Shared per-channel visibility ───────────────────────────────────────
    # One source of truth for the "Channels" checkboxes. Every panel that has
    # per-channel curves subscribes here: both hemisphere scatter plots AND both
    # polar slice plots, measured and fitted alike. These used to hang off the
    # *measured* scatter handle, so the fitted scatter and the fitted slice were
    # never notified and stayed frozen at their initial state (RGB on, lum off).
    channel_vis = (r   = Observable(true),
                   g   = Observable(true),
                   b   = Observable(true),
                   lum = Observable(false))

    # ── Reactive wiring helper: (scatter, surface, slice, data_obs) → live plots ──
    # The scatter and slice plots share `channel_vis`, so their R/G/B/Lum curves
    # toggle in lockstep with each other and with the opposite (measured/fitted)
    # panel.
    function _wire_plots!(scatter_h, surface_h, slice_h, data_obs)
        trig = Observable{Tuple{Union{Nothing, MeasuredBRDF}, Float32, Float32}}(
            (data_obs[], wi_theta[], wi_phi[])
        )
        fire() = (trig[] = (data_obs[], wi_theta[], wi_phi[]))
        on(_ -> fire(), data_obs)
        on(_ -> fire(), wi_theta)
        on(_ -> fire(), wi_phi)
        for key in (:r, :g, :b, :lum)
            scatter_h.channels[key].visible[] = channel_vis[key][]
            slice_h.channels[key].visible[]   = channel_vis[key][]
            on(channel_vis[key]) do enabled
                scatter_h.channels[key].visible[] = enabled
                slice_h.channels[key].visible[]   = enabled
                enabled && !isnothing(data_obs[]) && fire()
            end
        end
        on(_ -> !isnothing(data_obs[]) && fire(), scatter_h.n_points)
        on(_ -> !isnothing(data_obs[]) && fire(), surface_h.n_points)
        on(_ -> !isnothing(data_obs[]) && fire(), slice_h.n_samples)
        on(_ -> !isnothing(data_obs[]) && fire(), scatter_h.log_scale)
        on(_ -> !isnothing(data_obs[]) && fire(), surface_h.log_scale)
        on(_ -> !isnothing(data_obs[]) && fire(), slice_h.log_scale)

        on(trig) do (brdf, theta, phi)
            update_wi_indicator!(scatter_h, theta, phi)
            update_wi_indicator!(surface_h, theta, phi)
            update_slice_wi!(slice_h, theta)
            if isnothing(brdf)
                clear_hemisphere_plot!(scatter_h)
                clear_surface_plot!(surface_h)
                clear_slice_plot!(slice_h)
            else
                try
                    update_hemisphere_plot!(scatter_h, brdf, theta, phi)
                    update_surface_plot!(surface_h, brdf, theta, phi)
                    update_slice_plot!(slice_h, brdf, theta, phi)
                catch e
                    @error "Visualization error" exception=(e, catch_backtrace())
                end
            end
        end
    end

    _wire_plots!(scatter_handle,     surface_handle,     slice_handle,     loaded_brdf)
    _wire_plots!(scatter_handle_fit, surface_handle_fit, slice_handle_fit, fitted_brdf)

    # Incident-direction arrow visibility (synced across all 4 scientific plots)
    on(show_wi) do visible
        scatter_handle.show_wi[]     = visible
        surface_handle.show_wi[]     = visible
        scatter_handle_fit.show_wi[] = visible
        surface_handle_fit.show_wi[] = visible
    end

    # Link camera view (rotation + zoom) across all four scientific plots
    # (measured + fitted, scatter + surface) so dragging or scrolling on any one
    # moves them together.
    _sync_axis3_views!(scatter_handle.ax, surface_handle.ax,
                       scatter_handle_fit.ax, surface_handle_fit.ax)

    # Link the two polar slice plots (measured + fitted) the same way, so zoom,
    # pan and rotation carry across and both share one radial scale.
    _sync_polar_views!(slice_handle.ax, slice_handle_fit.ax)

    # Combined (theta, phi) observable so a single JS callback updates the
    # Three.js light direction whenever either slider moves.
    light_dir_obs = map(wi_theta, wi_phi) do t, p
        Float64[Float64(t), Float64(p)]
    end

    # ── Shared log scale observable (syncs all 4 plots) ─────────────────────
    log_scale_obs = Observable(false)
    on(log_scale_obs) do val
        scatter_handle.log_scale[]     = val
        surface_handle.log_scale[]     = val
        slice_handle.log_scale[]       = val
        scatter_handle_fit.log_scale[] = val
        surface_handle_fit.log_scale[] = val
        slice_handle_fit.log_scale[]   = val
    end

    # ── UI Controls ─────────────────────────────────────────────────────────

    function make_checkbox(channel_key::Symbol, checked::Bool)
        vis_obs = channel_vis[channel_key]
        DOM.input(
            type="checkbox",
            checked=checked,
            onchange=js"""function(event) { $(vis_obs).notify(event.target.checked); }""",
            style="margin-right: 3px;"
        )
    end

    # Log scale checkbox
    log_checkbox = DOM.input(
        type="checkbox",
        checked=false,
        onchange=js"""function(event) { $(log_scale_obs).notify(event.target.checked); }""",
        style="margin-right: 3px;"
    )

    input_style = "width: 80px; padding: 4px 6px; border: 1px solid #bdc3c7; border-radius: 3px; font-size: 0.85em;"
    label_style = "font-weight: bold; font-size: 0.85em; margin-right: 4px;"
    group_style = "display: inline-flex; align-items: center; margin-right: 16px;"
    select_style = "padding: 4px 6px; border: 1px solid #bdc3c7; border-radius: 3px; font-size: 0.85em; background: white;"

    # ── LUT resolution controls ─────────────────────────────────────────────
    res_input_style = "width: 56px; padding: 4px 6px; border: 1px solid #bdc3c7; border-radius: 3px; font-size: 0.85em;"
    _res_num(obs, val) = DOM.input(
        type = "number", value = val, min = 1, max = 128, step = 1,
        style = res_input_style,
        onchange = js"""function(event) {
            const v = parseInt(event.target.value);
            if (!isNaN(v)) $(obs).notify(v);
        }""",
    )
    resolution_controls = DOM.div(
        DOM.span("LUT res: ", style = label_style),
        DOM.span("θh", style = "font-size: 0.8em; margin: 0 2px;"), _res_num(res_th, res_th[]),
        DOM.span("θd", style = "font-size: 0.8em; margin: 0 2px 0 8px;"), _res_num(res_td, res_td[]),
        DOM.span("φd", style = "font-size: 0.8em; margin: 0 2px 0 8px;"), _res_num(res_pd, res_pd[]),
        DOM.span("φh", style = "font-size: 0.8em; margin: 0 2px 0 8px;"), _res_num(res_ph, res_ph[]),
        DOM.select(
            DOM.option("float16"; value = "float16", selected = res_dtype_pending[] == "float16"),
            DOM.option("float32"; value = "float32", selected = res_dtype_pending[] == "float32"),
            onchange = js"""function(event) { $(res_dtype_pending).notify(event.target.value); }""",
            style = "$select_style margin-left: 10px;",
        ),
        DOM.button(
            "Apply",
            onclick = js"""function(event) { $(res_apply_trigger).notify(Date.now()); }""",
            style = "padding: 4px 12px; margin-left: 10px; background: #16a085; color: white; " *
                    "border: none; border-radius: 3px; cursor: pointer; font-size: 0.85em; font-weight: bold;",
        ),
        res_info,
        DOM.span(res_status, style = "font-size: 0.8em; margin-left: 10px;"),
        style = "display: flex; align-items: center; flex-wrap: wrap; gap: 2px; margin-bottom: 8px;",
    )

    # ── Fit controls ────────────────────────────────────────────────────────
    model_select = DOM.select(
        [DOM.option(string(n); value=string(n),
                    selected = (string(n) == selected_model_name[])) for n in model_names]...,
        onchange=js"""function(event) { $(selected_model_name).notify(event.target.value); }""",
        style=select_style,
    )
    optimizer_select = DOM.select(
        [DOM.option(string(n); value=string(n),
                    selected = (string(n) == selected_optimizer_name[])) for n in optimizer_names]...,
        onchange=js"""function(event) { $(selected_optimizer_name).notify(event.target.value); }""",
        style=select_style,
    )

    # Error-map dropdown — picks which ErrorMap drives the "Error Map" sphere.
    # Option values are 1-based indices into AVAILABLE_ERROR_MAPS.
    errormap_select = DOM.select(
        [DOM.option(error_map_label(em); value=string(i), selected = (i == selected_errormap_idx[]))
         for (i, em) in enumerate(AVAILABLE_ERROR_MAPS)]...,
        onchange=js"""function(event) { $(selected_errormap_idx).notify(parseInt(event.target.value)); }""",
        style=select_style,
    )

    # ── Dynamic optimizer-option controls ──────────────────────────────────
    opt_value_obs = Ref(Dict{Symbol, Observable}())

    function _rebuild_option_obs!(name_sym::Symbol)
        new_dict = Dict{Symbol, Observable}()
        OptType = get(OPTIMIZER_REGISTRY, name_sym, nothing)
        if OptType !== nothing
            for spec in optimizer_options(OptType)
                new_dict[spec.name] = Observable{Any}(spec.default)
            end
        end
        opt_value_obs[] = new_dict
        return new_dict
    end
    _rebuild_option_obs!(Symbol(selected_optimizer_name[]))

    option_controls = map(selected_optimizer_name) do name
        d = _rebuild_option_obs!(Symbol(name))
        OptType = get(OPTIMIZER_REGISTRY, Symbol(name), nothing)
        OptType === nothing && return DOM.span()
        specs = optimizer_options(OptType)
        isempty(specs) && return DOM.span()
        elements = Any[]
        for spec in specs
            obs = d[spec.name]
            ctrl = if spec.kind === :choice
                DOM.select(
                    [DOM.option(string(c); value=string(c),
                                selected = (c == spec.default)) for c in spec.choices]...,
                    onchange=js"""function(event) { $(obs).notify(event.target.value); }""",
                    style=select_style,
                )
            elseif spec.kind === :int
                DOM.input(
                    type="number",
                    value=string(spec.default),
                    min=string(spec.range[1]),
                    max=string(spec.range[2]),
                    step=string(spec.step),
                    onchange=js"""function(event) { $(obs).notify(parseInt(event.target.value)); }""",
                    style=input_style,
                )
            elseif spec.kind === :float
                DOM.input(
                    type="number",
                    value=string(spec.default),
                    min=string(spec.range[1]),
                    max=string(spec.range[2]),
                    step=string(spec.step),
                    onchange=js"""function(event) { $(obs).notify(parseFloat(event.target.value)); }""",
                    style=input_style,
                )
            elseif spec.kind === :bool
                DOM.input(
                    type="checkbox",
                    checked=spec.default,
                    onchange=js"""function(event) { $(obs).notify(event.target.checked); }""",
                    style="margin-right: 3px;",
                )
            elseif spec.kind === :text
                DOM.input(
                    type="text",
                    value=string(spec.default),
                    onchange=js"""function(event) { $(obs).notify(event.target.value); }""",
                    style=input_style,
                )
            else
                DOM.span("(unsupported: $(spec.kind))")
            end
            push!(elements, DOM.span(
                DOM.span(spec.label * " "; style="font-size: 0.85em; margin: 0 4px 0 10px; color: #555;"),
                ctrl,
                title=spec.description,
                style="display: inline-flex; align-items: center;",
            ))
        end
        DOM.span(elements...)
    end

    fit_trigger = Observable(0)
    fit_button = DOM.button(
        "Fit",
        onclick=js"""function(event) { $(fit_trigger).notify(Date.now()); }""",
        style="padding: 6px 14px; background: #e67e22; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85em; font-weight: bold; width: fit-content;"
    )

    on(fit_trigger) do _
        brdf = loaded_brdf[]
        if isnothing(brdf)
            fit_status[] = "no_data"
            return
        end
        mkey = Symbol(selected_model_name[])
        okey = Symbol(selected_optimizer_name[])
        if !haskey(MODEL_REGISTRY, mkey) || !haskey(OPTIMIZER_REGISTRY, okey)
            fit_status[] = "error:unknown plugin"
            return
        end
        ModelType   = MODEL_REGISTRY[mkey]
        OptType     = OPTIMIZER_REGISTRY[okey]

        # Assemble kwargs from the introspected option observables.
        kwargs = Dict{Symbol, Any}()
        for spec in optimizer_options(OptType)
            haskey(opt_value_obs[], spec.name) || continue
            raw = opt_value_obs[][spec.name][]
            val = try
                if spec.kind === :choice
                    raw isa Symbol ? raw : Symbol(raw)
                elseif spec.kind === :int
                    Int(raw)
                elseif spec.kind === :float
                    Float64(raw)
                elseif spec.kind === :bool
                    Bool(raw)
                elseif spec.kind === :text
                    String(raw)
                else
                    raw
                end
            catch
                spec.default
            end
            kwargs[spec.name] = val
        end

        fit_status[] = "fitting"
        Threads.@spawn begin
            try
                opt = OptType(; kwargs...)
                t0  = time()
                m   = fit_brdf(opt, ModelType, brdf)
                dt  = round(time() - t0, digits=2)
                rmse = brdf_rmse(brdf, m)
                crwe = brdf_rms_crwe(brdf, m)
                rmsle = brdf_rmsle(brdf, m)
                ssim = image_ssim(brdf, m)
                fitted_model[] = m
                fit_rmse[] = rmse
                fit_crwe[] = crwe
                fit_rmsle[] = rmsle
                fit_ssim[] = ssim
                fit_status[] = "ok:$(dt)s"
                @info "Fit complete" model=ModelType seconds=dt rmse=rmse crwe=crwe rmsle=rmsle ssim_mean=ssim.mean ssim_min=ssim.min
            catch e
                @error "Fit failed" exception=(e, catch_backtrace())
                fit_status[] = "error:$(sprint(showerror, e))"
                fitted_model[] = nothing
                fit_rmse[] = nothing
                fit_crwe[] = nothing
                fit_ssim[] = nothing
                fit_rmsle[] = nothing
            end
        end
    end

    # Status badge
    status_badge = map(fit_status, fit_rmse, fit_crwe, fit_rmsle) do s, rmse, crwe, rmsle
        base = "padding: 4px 10px; border-radius: 3px; font-size: 0.85em; font-weight: bold; margin-left: 8px;"
        if s == "idle"
            DOM.span("idle", style=base * "background: #ecf0f1; color: #7f8c8d;")
        elseif s == "fitting"
            DOM.span("⏳ fitting…", style=base * "background: #fef5e7; color: #e67e22;")
        elseif s == "no_data"
            DOM.span("⚠ load a file first", style=base * "background: #fadbd8; color: #c0392b;")
        elseif startswith(s, "ok")
            rmsle_str = isnothing(rmsle) ? "" : "  RMS-LogWE=$(round(rmsle, sigdigits=4))"
            crwe_str = isnothing(crwe) ? "" : "  RMS-CRWE=$(round(crwe, sigdigits=4))"
            rmse_str = isnothing(rmse) ? "" : "  RMSE=$(round(rmse, sigdigits=4))"
            dt = split(s, ":")[2]
            DOM.span("✅ fitted in $(dt)$(rmsle_str)$(crwe_str)$(rmse_str)", style=base * "background: #d5f4e6; color: #27ae60;")
        else
            msg = length(s) > 80 ? s[1:77] * "…" : s
            DOM.span("❌ " * msg, style=base * "background: #fadbd8; color: #c0392b;")
        end
    end

    # Image-space quality badge: SSIM between CPU renders of measured vs fitted
    # (mean over the light directions in SSIM_LIGHT_DIRS; min = worst lighting).
    ssim_badge = map(fit_ssim) do ssim
        base = "padding: 4px 10px; border-radius: 3px; font-size: 0.85em; font-weight: bold; margin-left: 8px;"
        isnothing(ssim) && return DOM.span()
        txt = "🖼 SSIM=$(round(ssim.mean, digits=3)) (worst $(round(ssim.min, digits=3)))"
        style = if ssim.min >= 0.9
            base * "background: #d5f4e6; color: #27ae60;"
        elseif ssim.min >= 0.7
            base * "background: #fdebd0; color: #b9770e;"
        else
            base * "background: #fadbd8; color: #c0392b;"
        end
        DOM.span(txt, style=style)
    end

    # Parameters readout (uses display_parameters to cover every registered model)
    params_readout = map(fitted_model) do m
        if isnothing(m)
            return DOM.div("No fitted model.",
                           style="color: #7f8c8d; font-style: italic; font-size: 0.85em;")
        end
        try
            params = display_parameters(m)
            rows = []
            for (k, v) in params
                vstr = if v isa AbstractVector
                    "[" * join([string(round(Float64(x), sigdigits=5)) for x in v], ", ") * "]"
                elseif v isa Real
                    string(round(Float64(v), sigdigits=5))
                else
                    string(v)
                end
                push!(rows, DOM.div(
                    DOM.span(String(k); style="font-weight: bold; color: #34495e; margin-right: 8px;"),
                    DOM.span(vstr; style="font-family: monospace; color: #2c3e50;"),
                    style="margin-right: 18px; display: inline-block;"
                ))
            end
            DOM.div(rows...; style="display: flex; flex-wrap: wrap; gap: 6px;")
        catch e
            DOM.div("⚠ could not read parameters: $(sprint(showerror, e))",
                    style="color: #c0392b; font-size: 0.85em;")
        end
    end

    # ── Layout ──────────────────────────────────────────────────────────────
    return DOM.div(
        DOM.link(rel="stylesheet", href=APP_CSS),
        DOM.h1("BRDF Analysis Platform",
               style="color: #2c3e50; margin-bottom: 6px; font-size: 1.5em;"),
        DOM.hr(style="border: 1px solid #ecf0f1; margin-bottom: 12px;"),

        # ── Upper area: Two-column layout (Load BRDF | Fit Model) ────────
        DOM.div(
            # Left column: Load BRDF
            DOM.div(
                DOM.div(
                    DOM.span("📂 ", style="font-size: 1.1em;"),
                    DOM.span("Load BRDF: ", style="font-weight: bold; font-size: 0.9em; margin-right: 8px;"),
                    file_input_element,
                    DOM.span(" .binary / .bird / .json",
                              style="color: #7f8c8d; font-size: 0.8em; margin-left: 8px;"),
                    style="display: flex; align-items: center; flex-wrap: wrap; gap: 4px; margin-bottom: 6px;"
                ),
                # DOM.div(
                #     DOM.span("Or path: ", style="font-size: 0.85em; margin-right: 6px; white-space: nowrap;"),
                #     file_path_input,
                #     load_button,
                #     style="display: flex; align-items: center; gap: 6px;"
                # ),
                upload_indicator,
                status_display,
                style="padding: 10px; border: 1px dashed #bdc3c7; border-radius: 5px; flex: 1; min-width: 0;"
            ),
            # Right column: Fit Model
            DOM.div(
                DOM.div(
                    DOM.span("Fit model: ", style=label_style),
                    DOM.span("Model ", style="font-size: 0.85em; margin-right: 4px;"),
                    model_select,
                    DOM.span(" Optimizer ", style="font-size: 0.85em; margin: 0 4px 0 12px;"),
                    optimizer_select,
                    option_controls,
                    fit_button,
                    status_badge,
                    ssim_badge,
                    style="display: flex; align-items: center; flex-wrap: wrap; gap: 6px; margin-bottom: 6px;"
                ),
                DOM.div(
                    DOM.span("Parameters: ", style=label_style),
                    params_readout,
                    style="display: flex; align-items: center; flex-wrap: wrap; gap: 4px;"
                ),
                DOM.div(
                    DOM.span("Error map: ", style=label_style),
                    errormap_select,
                    style="display: flex; align-items: center; flex-wrap: wrap; gap: 4px; margin-top: 6px;"
                ),
                style="padding: 10px; background: #fdf6ec; border: 1px solid #f5cba7; border-radius: 5px; flex: 1; min-width: 0;"
            ),
            style="display: flex; gap: 10px; margin-bottom: 10px;"
        ),

        # ── All controls in one compact block ──────────────────────────────
        DOM.div(
            # Row 1: Angle sliders (side by side)
            DOM.div(
                DOM.div(
                    DOM.label("θ ", style=label_style),
                    DOM.div(theta_slider, style="flex: 1; min-width: 120px;"),
                    DOM.span(map(wi_theta) do t; " $(round(rad2deg(t), digits=1))°"; end,
                             style="min-width: 50px; color: #3498db; font-weight: bold; font-size: 0.85em; text-align: right;"),
                    style="display: flex; align-items: center; flex: 1; margin-right: 16px;"
                ),
                DOM.div(
                    DOM.label("φ ", style=label_style),
                    DOM.div(phi_slider, style="flex: 1; min-width: 120px;"),
                    DOM.span(map(wi_phi) do p; " $(round(rad2deg(p), digits=1))°"; end,
                             style="min-width: 50px; color: #3498db; font-weight: bold; font-size: 0.85em; text-align: right;"),
                    style="display: flex; align-items: center; flex: 1; margin-right: 16px;"
                ),
                DOM.div(
                    DOM.label("gain ", style=label_style),
                    DOM.div(gain_slider, style="flex: 1; min-width: 120px;"),
                    DOM.span(map(s -> " ×$(round(2.0^Float64(s), digits=2))", gain_slider.value),
                             style="min-width: 50px; color: #3498db; font-weight: bold; font-size: 0.85em; text-align: right;"),
                    style="display: flex; align-items: center; flex: 1;"
                ),
                style="display: flex; align-items: center; margin-bottom: 8px;"
            ),

            # Row 1b: Universal LUT resolution (θh, θd, φd, φh) + precision + Apply
            resolution_controls,

            # Row 2: Channels | Sample points (scatter & surface) | Log scale
            DOM.div(
                # Channel toggles
                DOM.div(
                    DOM.span("Channels: ", style=label_style),
                    (let spec = CHANNEL_SPECS[key]
                        DOM.label(
                            make_checkbox(key, key != :lum),
                            DOM.span("$(spec.label) ", style="color: $(spec.color_css); font-weight: bold; font-size: 0.85em; margin-right: 8px;"),
                            style="cursor: pointer; display: inline-flex; align-items: center;"
                        )
                    end for key in (:r, :g, :b, :lum))...,
                    style="$(group_style) margin-right: 20px;"
                ),

                # Scatter sample points
                DOM.div(
                    DOM.label("Scatter pts: ", style=label_style),
                    DOM.input(
                        type="number", value=scatter_handle.n_points[], min=50, max=10000, step=50,
                        style=input_style,
                        onchange=js"""function(event) {
                            const val = parseInt(event.target.value);
                            if (val >= 50 && val <= 10000) $(scatter_handle.n_points).notify(val);
                        }"""
                    ),
                    style=group_style
                ),

                # Surface sample points
                DOM.div(
                    DOM.label("Surface pts: ", style=label_style),
                    DOM.input(
                        type="number", value=surface_handle.n_points[], min=50, max=10000, step=50,
                        style=input_style,
                        onchange=js"""function(event) {
                            const val = parseInt(event.target.value);
                            if (val >= 50 && val <= 10000) $(surface_handle.n_points).notify(val);
                        }"""
                    ),
                    style=group_style
                ),

                # Log scale toggle
                DOM.div(
                    DOM.label(
                        log_checkbox,
                        DOM.span("Log scale", style="font-weight: bold; font-size: 0.85em;"),
                        style="cursor: pointer; display: inline-flex; align-items: center;"
                    ),
                    style=group_style
                ),

                style="display: flex; align-items: center; flex-wrap: wrap; gap: 4px;"
            ),

            # Row 3: Visibility toggles
            DOM.div(
                DOM.span("Show: ", style=label_style),
                DOM.label(
                    DOM.input(
                        type="checkbox",
                        checked=true,
                        onchange=js"""function(event) { $(show_scatter).notify(event.target.checked); }""",
                        style="margin-right: 3px;"
                    ),
                    DOM.span("Scatter", style="font-weight: bold; font-size: 0.85em; margin-right: 12px;"),
                    style="cursor: pointer; display: inline-flex; align-items: center;"
                ),
                DOM.label(
                    DOM.input(
                        type="checkbox",
                        checked=true,
                        onchange=js"""function(event) { $(show_surface).notify(event.target.checked); }""",
                        style="margin-right: 3px;"
                    ),
                    DOM.span("Surface", style="font-weight: bold; font-size: 0.85em; margin-right: 12px;"),
                    style="cursor: pointer; display: inline-flex; align-items: center;"
                ),
                DOM.label(
                    DOM.input(
                        type="checkbox",
                        checked=true,
                        onchange=js"""function(event) { $(show_slice).notify(event.target.checked); }""",
                        style="margin-right: 3px;"
                    ),
                    DOM.span("Slice", style="font-weight: bold; font-size: 0.85em; margin-right: 12px;"),
                    style="cursor: pointer; display: inline-flex; align-items: center;"
                ),
                DOM.label(
                    DOM.input(
                        type="checkbox",
                        checked=true,
                        onchange=js"""function(event) { $(show_render).notify(event.target.checked); }""",
                        style="margin-right: 3px;"
                    ),
                    DOM.span("Render", style="font-weight: bold; font-size: 0.85em; margin-right: 12px;"),
                    style="cursor: pointer; display: inline-flex; align-items: center;"
                ),
                DOM.label(
                    DOM.input(
                        type="checkbox",
                        checked=true,
                        onchange=js"""function(event) { $(show_wi).notify(event.target.checked); }""",
                        style="margin-right: 3px;"
                    ),
                    DOM.span("Incident Direction", style="font-weight: bold; font-size: 0.85em;"),
                    style="cursor: pointer; display: inline-flex; align-items: center;"
                ),
                # Render-object shape selector (applies to all three render panels)
                DOM.div(
                    DOM.label("Shape: ", style="$label_style margin-left: 16px;"),
                    DOM.select(
                        [DOM.option(label; value=val, selected = (val == render_shape[]))
                         for (val, label) in SHAPE_LABELS]...,
                        onchange=js"""function(event) { $(render_shape).notify(event.target.value); }""",
                        style=select_style,
                    ),
                    style="display: inline-flex; align-items: center;"
                ),
                style="display: flex; align-items: center; flex-wrap: wrap; gap: 4px; padding-top: 8px; border-top: 1px solid #d0d3d4;"
            ),

            # Sticky so the controls (incident direction, gain, shape, channels…)
            # stay in reach while scrolling through the plots below. The opaque
            # background hides content scrolling underneath; the shadow separates
            # it from the plots once stuck. (The nav bar is no longer sticky — see
            # _nav_bar in hub.jl.)
            style="padding: 10px; background: #ecf0f1; border-radius: 5px; margin-bottom: 10px; " *
                  "position: sticky; top: 0; z-index: 50; box-shadow: 0 2px 6px rgba(0,0,0,0.12);"
        ),

        # -- Visualizations -- 3-column grid (.grid CSS lays sections side by side:
        #    h3 titles -> row 1, bodies -> row 2). Each section's (h3, body) tuple is
        #    splatted in so they stay flat siblings of `.grid`. See viz_blocks.jl.
        DOM.div(
            viz_section("Measured",
                scatter_block(fig_scatter, scatter_handle, show_scatter),
                surface_block(fig_surface, surface_handle, show_surface),
                slice_block(fig_slice, slice_handle, show_slice),
                render_sphere_block(session, measured_render_payload, light_dir_obs, show_render;
                                    renderer_module=UNIVERSAL_RENDERER, global_key="_brdfMeasured",
                                    subtitle=" · Three.js · Data-Driven", margin="margin-left: 6px;",
                                    render_shape_obs=render_shape, intensity_obs=render_gain_obs,
                                    block_title=shape_title, show_wi_obs=show_wi);
                reverse=true,
                title_style="color: #2c3e50; margin: 6px 0 4px 0; font-size: 1em;")...,

            viz_section("Fitted Model",
                scatter_block(fig_scatter_fit, scatter_handle_fit, show_scatter),
                surface_block(fig_surface_fit, surface_handle_fit, show_surface),
                slice_block(fig_slice_fit, slice_handle_fit, show_slice),
                render_sphere_block(session, fitted_render_obs, light_dir_obs, show_render;
                                    renderer_module=UNIVERSAL_RENDERER, global_key="_brdfFitted",
                                    subtitle=" · Three.js · Analytical", margin="margin-left: 6px;",
                                    render_shape_obs=render_shape, intensity_obs=render_gain_obs,
                                    block_title=shape_title, show_wi_obs=show_wi);
                reverse=true)...,

            viz_section("Error Map",
                render_sphere_block(session, error_render_obs, light_dir_obs, show_render;
                                    renderer_module=UNIVERSAL_RENDERER, global_key="_brdfError",
                                    subtitle=map(em -> " · measured vs fitted · $(error_map_label(em))",
                                                 selected_errormap),
                                    render_shape_obs=render_shape,
                                    legend_obs=map(error_map_legend, selected_errormap),
                                    block_title=shape_title, show_wi_obs=show_wi))...,
        class="grid"),

        style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 20px; max-width: 1600px; margin: 0 auto;"
    )
end

"""
    run_interactive_app(; port=8000, host="0.0.0.0", proxy_url=nothing)

Start the Bonito web server.

When `host="0.0.0.0"`, `proxy_url` is auto-detected as `http://<LAN-IP>:<port>` so
the browser fetches Bonito's runtime/assets/websocket from a routable address
instead of `0.0.0.0` (which no browser can reach — the page would never bootstrap,
even via `127.0.0.1`). Pass `proxy_url` explicitly to override (e.g. reverse proxy).
"""
function run_interactive_app(; port=8000, host="0.0.0.0", proxy_url=nothing)
    app = App(create_interactive_app)
    purl = resolve_proxy_url(proxy_url, host, port)

    println("=" ^ 60)
    println("BRDF Analysis Platform")
    println("=" ^ 60)
    println("local:  http://$host:$port")
    hint = remote_url_hint(purl, port)
    isempty(hint) || println("remote: $hint")
    println("$(Threads.nthreads()) thread(s)")
    println("Ctrl+C to stop")
    println("=" ^ 60)

    return Bonito.Server(app, host, port; proxy_url=purl)
end

export create_interactive_app, run_interactive_app