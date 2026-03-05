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

    # ── File upload handling ────────────────────────────────────────────────
    file_input_element = DOM.input(
        type="file",
        accept=".binary,.bird,.json",
        id="brdf-file-input",
        style="padding: 6px; border: 1px solid #bdc3c7; border-radius: 4px; cursor: pointer; font-size: 0.85em;",
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
                @info "Loaded uploaded file: $(data.name)"
            else
                @warn "Failed to load BRDF: $(data.error_msg)"
                loaded_brdf[] = nothing
                upload_status[] = "error"
            end
        catch e
            @error "Error processing upload" exception=(e, catch_backtrace())
            loaded_brdf[] = nothing
            upload_status[] = "error"
        end
    end

    # Manual path entry
    file_path_input = DOM.input(
        type="text",
        placeholder="/path/to/file.binary",
        style="flex: 1; padding: 6px; border: 1px solid #ccc; border-radius: 4px; font-size: 0.85em;"
    )
    load_button = DOM.button(
        "Load",
        onclick=js"""function(event) {
            const path = $(file_path_input).value;
            if (path) $(file_path_obs).notify(path);
        }""",
        style="padding: 6px 14px; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85em;"
    )

    on(file_path_obs) do filepath
        try
            (isnothing(filepath) || isempty(filepath)) && return
            !isfile(filepath) && (@warn "File not found: $filepath"; return)
            data = load_brdf(filepath)
            if data.status == :ok
                loaded_brdf[] = data
                @info "Loaded: $(data.name)"
            else
                @warn "Failed to load BRDF: $(data.error_msg)"
                loaded_brdf[] = nothing
            end
        catch e
            @error "Error loading file" exception=(e, catch_backtrace())
            loaded_brdf[] = nothing
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
            DOM.div(msg, style="color: $fg; padding: 6px 10px; background: $bg; border-left: 3px solid $fg; border-radius: 3px; margin-top: 6px; font-size: 0.85em;")
        else
            DOM.div(style="display: none;")
        end
    end

    # ── Loaded file status (compact) ────────────────────────────────────────
    status_display = map(loaded_brdf) do brdf
        if isnothing(brdf)
            return DOM.div("📁 No file loaded.", style="color: #7f8c8d; padding: 6px; font-style: italic; font-size: 0.85em;")
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

    # ── Hemisphere scatter plot (via plots.jl) ────────────────────────────────
    fig_scatter = Figure(resolution=(700, 700))
    scatter_handle = setup_hemisphere_plot!(fig_scatter[1, 1];
                                            n_points=800,
                                            show_r=true, show_g=true, show_b=true, show_lum=false)

    # ── Surface plot (via plots.jl) — now Fibonacci-sampled mesh ────────────
    fig_surface = Figure(resolution=(700, 700))
    surface_handle = setup_surface_plot!(fig_surface[1, 1]; n_points=800)

    # ── Reactive wiring: BRDF / angles → both plots ────────────────────────
    _trigger = Observable{Tuple{Union{Nothing, MeasuredBRDF}, Float32, Float32}}(
        (nothing, wi_theta[], wi_phi[])
    )
    on(loaded_brdf) do b; _trigger[] = (b, wi_theta[], wi_phi[]); end
    on(wi_theta)    do t; _trigger[] = (loaded_brdf[], t, wi_phi[]); end
    on(wi_phi)      do p; _trigger[] = (loaded_brdf[], wi_theta[], p); end

    # Recalculate when a channel is toggled on
    for key in (:r, :g, :b, :lum)
        on(scatter_handle.channels[key].visible) do enabled
            if enabled && !isnothing(loaded_brdf[])
                _trigger[] = (loaded_brdf[], wi_theta[], wi_phi[])
            end
        end
    end

    # Recalculate when scatter n_points changes
    on(scatter_handle.n_points) do _
        if !isnothing(loaded_brdf[])
            _trigger[] = (loaded_brdf[], wi_theta[], wi_phi[])
        end
    end

    # Recalculate when surface n_points changes
    on(surface_handle.n_points) do _
        if !isnothing(loaded_brdf[])
            _trigger[] = (loaded_brdf[], wi_theta[], wi_phi[])
        end
    end

    # Recalculate when log scale changes
    on(scatter_handle.log_scale) do _
        if !isnothing(loaded_brdf[])
            _trigger[] = (loaded_brdf[], wi_theta[], wi_phi[])
        end
    end
    on(surface_handle.log_scale) do _
        if !isnothing(loaded_brdf[])
            _trigger[] = (loaded_brdf[], wi_theta[], wi_phi[])
        end
    end

    on(_trigger) do (brdf, theta, phi)
        if isnothing(brdf)
            clear_hemisphere_plot!(scatter_handle)
            clear_surface_plot!(surface_handle)
        else
            try
                update_hemisphere_plot!(scatter_handle, brdf, theta, phi)
                update_surface_plot!(surface_handle, brdf, theta, phi)
            catch e
                @error "Visualization error" exception=(e, catch_backtrace())
            end
        end
    end

    # ── Shared log scale observable (syncs both plots) ──────────────────────
    log_scale_obs = Observable(false)
    on(log_scale_obs) do val
        scatter_handle.log_scale[] = val
        surface_handle.log_scale[] = val
    end

    # ── UI Controls ─────────────────────────────────────────────────────────

    # Channel toggle checkboxes (scatter plot)
    function make_checkbox(channel_key::Symbol, checked::Bool)
        vis_obs = scatter_handle.channels[channel_key].visible
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

    # Compact control style constants
    input_style = "width: 80px; padding: 4px 6px; border: 1px solid #bdc3c7; border-radius: 3px; font-size: 0.85em;"
    label_style = "font-weight: bold; font-size: 0.85em; margin-right: 4px;"
    group_style = "display: inline-flex; align-items: center; margin-right: 16px;"

    # ── Layout ──────────────────────────────────────────────────────────────
    return DOM.div(
        DOM.h1("🔬 BRDF Analysis Platform",
               style="color: #2c3e50; margin-bottom: 6px; font-size: 1.5em;"),
        DOM.hr(style="border: 1px solid #ecf0f1; margin-bottom: 12px;"),

        # ── File upload (compact) ───────────────────────────────────────────
        DOM.div(
            DOM.div(
                DOM.span("📂 ", style="font-size: 1.1em;"),
                DOM.span("Load BRDF: ", style="font-weight: bold; font-size: 0.9em; margin-right: 8px;"),
                file_input_element,
                DOM.span(" .binary / .bird / .json",
                          style="color: #7f8c8d; font-size: 0.8em; margin-left: 8px;"),
                style="display: flex; align-items: center; flex-wrap: wrap; gap: 4px; margin-bottom: 6px;"
            ),
            DOM.div(
                DOM.span("Or path: ", style="font-size: 0.85em; margin-right: 6px; white-space: nowrap;"),
                file_path_input,
                load_button,
                style="display: flex; align-items: center; gap: 6px;"
            ),
            upload_indicator,
            status_display,
            style="padding: 10px; border: 1px dashed #bdc3c7; border-radius: 5px; margin-bottom: 10px;"
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
                    style="display: flex; align-items: center; flex: 1;"
                ),
                style="display: flex; align-items: center; margin-bottom: 8px;"
            ),

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

            style="padding: 10px; background: #ecf0f1; border-radius: 5px; margin-bottom: 10px;"
        ),

        # ── Visualizations — side by side ──────────────────────────────────
        DOM.div(
            # Scatter plot
            DOM.div(
                DOM.div(
                    DOM.h4("Scatter (per-channel)", style="color: #34495e; margin: 0; display: inline;"),
                    DOM.span(map(scatter_handle.n_points) do n; " · $(n) pts"; end,
                              style="color: #7f8c8d; font-size: 0.8em; font-style: italic;"),
                    style="margin-bottom: 4px;"
                ),
                DOM.div(fig_scatter, style="background: white; border: 1px solid #dee2e6; border-radius: 5px; padding: 6px;"),
                style="flex: 1; min-width: 0; margin-right: 6px;"
            ),
            # Surface plot
            DOM.div(
                DOM.div(
                    DOM.h4("Surface (luminance shape, RGB color)", style="color: #34495e; margin: 0; display: inline;"),
                    DOM.span(map(surface_handle.n_points) do n; " · $(n) pts"; end,
                              style="color: #7f8c8d; font-size: 0.8em; font-style: italic;"),
                    style="margin-bottom: 4px;"
                ),
                DOM.div(fig_surface, style="background: white; border: 1px solid #dee2e6; border-radius: 5px; padding: 6px;"),
                style="flex: 1; min-width: 0; margin-left: 6px;"
            ),
            style="display: flex; gap: 6px;"
        ),

        DOM.hr(style="border: 1px solid #ecf0f1; margin-top: 20px; margin-bottom: 10px;"),
        DOM.p("BRDF Analysis Platform • 2025",
              style="text-align: center; color: #95a5a6; font-size: 0.85em;"),

        style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 20px; max-width: 1600px; margin: 0 auto;"
    )
end

"""
    run_interactive_app(; port=8000, host="127.0.0.1")

Start the Bonito web server.
"""
function run_interactive_app(; port=8000, host="127.0.0.1")
    app = App(create_interactive_app)

    println("=" ^ 60)
    println("🚀 BRDF Analysis Platform")
    println("=" ^ 60)
    println("📡 http://$host:$port")
    println("📊 $(Threads.nthreads()) thread(s)")
    println("⏹️  Ctrl+C to stop")
    println("=" ^ 60)

    return Bonito.Server(app, host, port)
end

export create_interactive_app, run_interactive_app