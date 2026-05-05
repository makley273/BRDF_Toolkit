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

# ── Three.js renderer module (measured MERL BRDF) ─────────────────────────
# Self-contained ES6 module that imports Three.js + OrbitControls from the
# CDN and exports `createMerlRenderer(container)`. Loaded lazily by Bonito.
const MERL_RENDERER = ES6Module(joinpath(@__DIR__, "..", "rendering", "js",
                                          "merl_renderer.js"))

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
                     wo::SVector{3, Float32})::RGB{Float32}
    return evaluate(w.model, wi, wo)
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
Compute RMSE (luminance) between measured BRDF and fitted model over a
deterministic Fibonacci lattice of (wi, wo) pairs. Purely diagnostic.
"""
function _compute_rmse(measured::MeasuredBRDF, model::BRDFModel; n::Int=1024)
    g1 = 0.7548776662466927
    g2 = 0.5698402909980532
    sum_sq = 0.0
    count = 0
    for k in 1:n
        u1 = mod(k * g1, 1.0); u2 = mod(k * g2, 1.0)
        v1 = mod((k + 0.5) * g1 + 0.3, 1.0); v2 = mod((k + 0.5) * g2 + 0.7, 1.0)
        r  = sqrt(u1);  phi_i = 2π * u2
        r2 = sqrt(v1);  phi_o = 2π * v2
        wi = SVector{3, Float32}(r * cos(phi_i), r * sin(phi_i),  sqrt(max(0.0, 1.0 - u1)))
        wo = SVector{3, Float32}(r2 * cos(phi_o), r2 * sin(phi_o), sqrt(max(0.0, 1.0 - v1)))
        cm = lookup(measured, wi, wo)
        cf = evaluate(model, wi, wo)
        lm_ = 0.299f0 * cm.r + 0.587f0 * cm.g + 0.114f0 * cm.b
        lf  = 0.299f0 * cf.r + 0.587f0 * cf.g + 0.114f0 * cf.b
        sum_sq += (Float64(lm_) - Float64(lf))^2
        count += 1
    end
    return sqrt(sum_sq / max(count, 1))
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
    
    # Visibility toggles for visualization panels
    show_scatter = Observable{Bool}(true)
    show_surface = Observable{Bool}(true)
    show_render = Observable{Bool}(true)

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
                # Push raw bytes to the JS renderer when format has one
                merl_bytes_b64[] = data.source_format == :merl ? file_data["data"] : nothing
                @info "Loaded uploaded file: $(data.name)"
            else
                @warn "Failed to load BRDF: $(data.error_msg)"
                loaded_brdf[] = nothing
                merl_bytes_b64[] = nothing
                upload_status[] = "error"
            end
        catch e
            @error "Error processing upload" exception=(e, catch_backtrace())
            loaded_brdf[] = nothing
            merl_bytes_b64[] = nothing
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
                # Push raw bytes to the JS renderer when format has one
                merl_bytes_b64[] = data.source_format == :merl ?
                                   base64encode(read(filepath)) : nothing
                @info "Loaded: $(data.name)"
            else
                @warn "Failed to load BRDF: $(data.error_msg)"
                loaded_brdf[] = nothing
                merl_bytes_b64[] = nothing
            end
        catch e
            @error "Error loading file" exception=(e, catch_backtrace())
            loaded_brdf[] = nothing
            merl_bytes_b64[] = nothing
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

    # ── Hemisphere scatter + surface plots: measured ───────────────────────
    fig_scatter = Figure(resolution=(700, 700))
    scatter_handle = setup_hemisphere_plot!(fig_scatter[1, 1];
                                            n_points=800,
                                            show_r=true, show_g=true, show_b=true, show_lum=false)

    fig_surface = Figure(resolution=(700, 700))
    surface_handle = setup_surface_plot!(fig_surface[1, 1]; n_points=800)

    # ── Hemisphere scatter + surface plots: fitted model ───────────────────
    fig_scatter_fit = Figure(resolution=(700, 700))
    scatter_handle_fit = setup_hemisphere_plot!(fig_scatter_fit[1, 1];
                                                n_points=800,
                                                show_r=true, show_g=true, show_b=true, show_lum=false)

    fig_surface_fit = Figure(resolution=(700, 700))
    surface_handle_fit = setup_surface_plot!(fig_surface_fit[1, 1]; n_points=800)

    # ── Three.js renderer state (measured panel only for now) ──────────────
    # Per-format renderers will be added later. For each measured format we
    # ship the raw file bytes (base64) to the renderer module on the JS side;
    # the module decodes, builds a 3D texture, and swaps the sphere material.
    # `nothing` clears the renderer / leaves the placeholder material.
    merl_bytes_b64 = Observable{Union{Nothing, String}}(nothing)

    # ── Fit state ──────────────────────────────────────────────────────────
    model_names     = sort(collect(keys(MODEL_REGISTRY)))
    optimizer_names = sort(collect(keys(OPTIMIZER_REGISTRY)))
    selected_model_name     = Observable(string(isempty(model_names) ? "" : model_names[1]))
    selected_optimizer_name = Observable(string(isempty(optimizer_names) ? "" : optimizer_names[1]))
    fitted_model  = Observable{Union{Nothing, BRDFModel}}(nothing)
    fitted_brdf   = Observable{Union{Nothing, MeasuredBRDF}}(nothing)
    fit_status    = Observable{String}("idle")
    fit_rmse      = Observable{Union{Nothing, Float64}}(nothing)

    # When a new model is fitted, wrap it for the plot handles
    on(fitted_model) do m
        if isnothing(m)
            fitted_brdf[] = nothing
        else
            source = isnothing(loaded_brdf[]) ? "unknown" : loaded_brdf[].name
            wls    = isnothing(loaded_brdf[]) ? Float32[630, 532, 465] : loaded_brdf[].wavelengths
            fitted_brdf[] = _model_to_measured(m, wls, source)
        end
    end

    # Invalidate fit when the underlying data changes
    on(loaded_brdf) do _
        fitted_model[] = nothing
        fit_status[] = "idle"
        fit_rmse[] = nothing
    end

    # ── Reactive wiring helper: (scatter, surface, data_obs) → live plots ──
    function _wire_plots!(scatter_h, surface_h, data_obs)
        trig = Observable{Tuple{Union{Nothing, MeasuredBRDF}, Float32, Float32}}(
            (data_obs[], wi_theta[], wi_phi[])
        )
        fire() = (trig[] = (data_obs[], wi_theta[], wi_phi[]))
        on(_ -> fire(), data_obs)
        on(_ -> fire(), wi_theta)
        on(_ -> fire(), wi_phi)
        for key in (:r, :g, :b, :lum)
            on(scatter_h.channels[key].visible) do enabled
                enabled && !isnothing(data_obs[]) && fire()
            end
        end
        on(_ -> !isnothing(data_obs[]) && fire(), scatter_h.n_points)
        on(_ -> !isnothing(data_obs[]) && fire(), surface_h.n_points)
        on(_ -> !isnothing(data_obs[]) && fire(), scatter_h.log_scale)
        on(_ -> !isnothing(data_obs[]) && fire(), surface_h.log_scale)

        on(trig) do (brdf, theta, phi)
            if isnothing(brdf)
                clear_hemisphere_plot!(scatter_h)
                clear_surface_plot!(surface_h)
            else
                try
                    update_hemisphere_plot!(scatter_h, brdf, theta, phi)
                    update_surface_plot!(surface_h, brdf, theta, phi)
                catch e
                    @error "Visualization error" exception=(e, catch_backtrace())
                end
            end
        end
    end

    _wire_plots!(scatter_handle,     surface_handle,     loaded_brdf)
    _wire_plots!(scatter_handle_fit, surface_handle_fit, fitted_brdf)

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
        scatter_handle_fit.log_scale[] = val
        surface_handle_fit.log_scale[] = val
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
    select_style = "padding: 4px 6px; border: 1px solid #bdc3c7; border-radius: 3px; font-size: 0.85em; background: white;"

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

    fit_trigger = Observable(0)
    fit_button = DOM.button(
        "🎯 Fit",
        onclick=js"""function(event) { $(fit_trigger).notify(Date.now()); }""",
        style="padding: 6px 14px; background: #e67e22; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85em; font-weight: bold;"
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

        fit_status[] = "fitting"
        Threads.@spawn begin
            try
                opt = OptType()                       # default-constructed optimizer
                t0  = time()
                m   = fit_brdf(opt, ModelType, brdf)
                dt  = round(time() - t0, digits=2)
                rmse = _compute_rmse(brdf, m)
                fitted_model[] = m
                fit_rmse[] = rmse
                fit_status[] = "ok:$(dt)s"
                @info "Fit complete" model=ModelType seconds=dt rmse=rmse
            catch e
                @error "Fit failed" exception=(e, catch_backtrace())
                fit_status[] = "error:$(sprint(showerror, e))"
                fitted_model[] = nothing
                fit_rmse[] = nothing
            end
        end
    end

    # Status badge
    status_badge = map(fit_status, fit_rmse) do s, rmse
        base = "padding: 4px 10px; border-radius: 3px; font-size: 0.85em; font-weight: bold; margin-left: 8px;"
        if s == "idle"
            DOM.span("idle", style=base * "background: #ecf0f1; color: #7f8c8d;")
        elseif s == "fitting"
            DOM.span("⏳ fitting…", style=base * "background: #fef5e7; color: #e67e22;")
        elseif s == "no_data"
            DOM.span("⚠ load a file first", style=base * "background: #fadbd8; color: #c0392b;")
        elseif startswith(s, "ok")
            rmse_str = isnothing(rmse) ? "" : "  RMSE=$(round(rmse, sigdigits=4))"
            dt = split(s, ":")[2]
            DOM.span("✅ fitted in $(dt)$(rmse_str)", style=base * "background: #d5f4e6; color: #27ae60;")
        else
            msg = length(s) > 80 ? s[1:77] * "…" : s
            DOM.span("❌ " * msg, style=base * "background: #fadbd8; color: #c0392b;")
        end
    end

    # Parameters readout (uses shader_parameters to cover every registered model)
    params_readout = map(fitted_model) do m
        if isnothing(m)
            return DOM.div("No fitted model.",
                           style="color: #7f8c8d; font-style: italic; font-size: 0.85em;")
        end
        try
            params = shader_parameters(m)
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
        DOM.h1("🔬 BRDF Analysis Platform",
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
                DOM.div(
                    DOM.span("Or path: ", style="font-size: 0.85em; margin-right: 6px; white-space: nowrap;"),
                    file_path_input,
                    load_button,
                    style="display: flex; align-items: center; gap: 6px;"
                ),
                upload_indicator,
                status_display,
                style="padding: 10px; border: 1px dashed #bdc3c7; border-radius: 5px; flex: 1; min-width: 0;"
            ),
            # Right column: Fit Model
            DOM.div(
                DOM.div(
                    DOM.span("🎯 Fit model: ", style=label_style),
                    DOM.span("Model ", style="font-size: 0.85em; margin-right: 4px;"),
                    model_select,
                    DOM.span(" Optimizer ", style="font-size: 0.85em; margin: 0 4px 0 12px;"),
                    optimizer_select,
                    fit_button,
                    status_badge,
                    style="display: flex; align-items: center; flex-wrap: wrap; gap: 6px; margin-bottom: 6px;"
                ),
                DOM.div(
                    DOM.span("Parameters: ", style=label_style),
                    params_readout,
                    style="display: flex; align-items: center; flex-wrap: wrap; gap: 4px;"
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
                        onchange=js"""function(event) { $(show_render).notify(event.target.checked); }""",
                        style="margin-right: 3px;"
                    ),
                    DOM.span("Render", style="font-weight: bold; font-size: 0.85em;"),
                    style="cursor: pointer; display: inline-flex; align-items: center;"
                ),
                style="display: flex; align-items: center; flex-wrap: wrap; gap: 4px; padding-top: 8px; border-top: 1px solid #d0d3d4;"
            ),

            style="padding: 10px; background: #ecf0f1; border-radius: 5px; margin-bottom: 10px;"
        ),

        # ── Visualizations — 2x3 grid with visibility controls ─────────────
        DOM.div(
            DOM.h3("Measured", style="color: #2c3e50; margin: 6px 0 4px 0; font-size: 1em;"),
            DOM.div(
                # Scatter plot (with visibility toggle)
                map(show_scatter) do visible
                    if !visible
                        return DOM.div(style="display: none;")
                    end
                    DOM.div(
                        DOM.div(
                            DOM.h4("Scatter (per-channel)", style="color: #34495e; margin: 0; display: inline;"),
                            DOM.span(map(scatter_handle.n_points) do n; " · $(n) pts"; end,
                                      style="color: #7f8c8d; font-size: 0.8em; font-style: italic;"),
                            style="margin-bottom: 4px;"
                        ),
                        DOM.div(fig_scatter, style="background: white; border: 1px solid #dee2e6; border-radius: 5px; padding: 6px;"),
                        style="flex: 1; min-width: 0; margin-right: 6px;"
                    )
                end,
                # Surface plot (with visibility toggle)
                map(show_surface) do visible
                    if !visible
                        return DOM.div(style="display: none;")
                    end
                    DOM.div(
                        DOM.div(
                            DOM.h4("Surface (luminance shape, RGB color)", style="color: #34495e; margin: 0; display: inline;"),
                            DOM.span(map(surface_handle.n_points) do n; " · $(n) pts"; end,
                                      style="color: #7f8c8d; font-size: 0.8em; font-style: italic;"),
                            style="margin-bottom: 4px;"
                        ),
                        DOM.div(fig_surface, style="background: white; border: 1px solid #dee2e6; border-radius: 5px; padding: 6px;"),
                        style="flex: 1; min-width: 0; margin: 0 6px;"
                    )
                end,
                # Render (measured) — Three.js MERL renderer
                map(show_render) do visible
                    if !visible
                        return DOM.div(style="display: none;")
                    end
                    # Per-element canvas container; bootstrapped via onload.
                    canvas_container = DOM.div(
                        style = "width: 100%; height: 400px; background: #202020; " *
                                "border-radius: 5px; overflow: hidden; position: relative;"
                    )
                    Bonito.onload(session, canvas_container, js"""
                        function (el) {
                            $(MERL_RENDERER).then(mod => {
                                const r = mod.createMerlRenderer(el);
                                el._brdfRenderer = r;
                                window._brdfMeasured = r;
                                const dir = $(light_dir_obs);
                                r.setLightDir(dir[0], dir[1]);
                                const b64 = $(merl_bytes_b64);
                                if (b64) r.loadBufferBase64(b64);
                            }).catch(err => {
                                console.error('[merl_renderer] load failed:', err);
                            });
                        }
                    """)
                    Bonito.onjs(session, merl_bytes_b64, js"""
                        function (b64) {
                            const r = window._brdfMeasured;
                            if (r && b64) r.loadBufferBase64(b64);
                        }
                    """)
                    Bonito.onjs(session, light_dir_obs, js"""
                        function (dir) {
                            const r = window._brdfMeasured;
                            if (r) r.setLightDir(dir[0], dir[1]);
                        }
                    """)
                    DOM.div(
                        DOM.div(
                            DOM.h4("Sphere Render", style="color: #34495e; margin: 0; display: inline;"),
                            DOM.span(" · Three.js · MERL only",
                                      style="color: #7f8c8d; font-size: 0.8em; font-style: italic;"),
                            style="margin-bottom: 4px;"
                        ),
                        DOM.div(canvas_container,
                                style="background: white; border: 1px solid #dee2e6; border-radius: 5px; padding: 6px;"),
                        style="flex: 1; min-width: 0; margin-left: 6px;"
                    )
                end,
                style="display: flex; gap: 6px;"
            ),

            DOM.h3("Fitted Model", style="color: #2c3e50; margin: 12px 0 4px 0; font-size: 1em;"),
            DOM.div(
                # Scatter plot (with visibility toggle)
                map(show_scatter) do visible
                    if !visible
                        return DOM.div(style="display: none;")
                    end
                    DOM.div(
                        DOM.div(
                            DOM.h4("Scatter (per-channel)", style="color: #34495e; margin: 0; display: inline;"),
                            DOM.span(map(scatter_handle_fit.n_points) do n; " · $(n) pts"; end,
                                      style="color: #7f8c8d; font-size: 0.8em; font-style: italic;"),
                            style="margin-bottom: 4px;"
                        ),
                        DOM.div(fig_scatter_fit, style="background: white; border: 1px solid #dee2e6; border-radius: 5px; padding: 6px;"),
                        style="flex: 1; min-width: 0; margin-right: 6px;"
                    )
                end,
                # Surface plot (with visibility toggle)
                map(show_surface) do visible
                    if !visible
                        return DOM.div(style="display: none;")
                    end
                    DOM.div(
                        DOM.div(
                            DOM.h4("Surface (luminance shape, RGB color)", style="color: #34495e; margin: 0; display: inline;"),
                            DOM.span(map(surface_handle_fit.n_points) do n; " · $(n) pts"; end,
                                      style="color: #7f8c8d; font-size: 0.8em; font-style: italic;"),
                            style="margin-bottom: 4px;"
                        ),
                        DOM.div(fig_surface_fit, style="background: white; border: 1px solid #dee2e6; border-radius: 5px; padding: 6px;"),
                        style="flex: 1; min-width: 0; margin: 0 6px;"
                    )
                end,
                # Render (fitted) — placeholder; per-model renderers coming
                map(show_render, fitted_model) do visible, m
                    if !visible
                        return DOM.div(style="display: none;")
                    end
                    label = isnothing(m) ? "(no model fitted yet)" :
                                            "Renderer for $(nameof(typeof(m))) not yet implemented"
                    DOM.div(
                        DOM.div(
                            DOM.h4("Sphere Render", style="color: #34495e; margin: 0; display: inline;"),
                            DOM.span(" · per-model renderer pending",
                                      style="color: #7f8c8d; font-size: 0.8em; font-style: italic;"),
                            style="margin-bottom: 4px;"
                        ),
                        DOM.div(
                            DOM.div(
                                DOM.p("🚧 " * label,
                                      style="text-align: center; color: #7f8c8d; font-size: 0.95em; margin-top: 60px;"),
                                DOM.p("A LUT-based fallback and per-model GLSL renderers will be added later.",
                                      style="text-align: center; color: #95a5a6; font-size: 0.8em; margin-top: 8px;"),
                                style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%;"
                            ),
                            style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; padding: 6px; min-height: 400px;"
                        ),
                        style="flex: 1; min-width: 0; margin-left: 6px;"
                    )
                end,
                style="display: flex; gap: 6px;"
            ),
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