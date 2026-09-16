"""
Modular visualization blocks for the interactive app.

`interactive_app.jl` used to inline three near-identical ~50–75-line DOM blocks
(Measured / Fitted / Error spheres) plus repeated scatter/surface plot markup.
These functions return those DOM fragments so the app file stays pure layout +
reactive wiring.

All state is passed in explicitly — these are module-level functions and cannot
see `create_interactive_app`'s locals. In particular the Three.js renderer module
(`UNIVERSAL_RENDERER`) and the per-sphere `window[...]` global key are arguments,
so the three spheres never clobber each other's renderer.

Layout note: `render_sphere_block` / `*_block` return single DOM nodes, but
`viz_section` returns a **tuple** `(h3, body)` of *siblings* — the parent uses the
`.grid` CSS (`:nth-child` odd→row1, even→row2) to lay the three sections out as
columns, so the section title and body must remain flat siblings of `.grid`, not
nested in a wrapper. Splat the tuple into the parent `DOM.div(...)`.

Location: src/app/viz_blocks.jl
"""

using Bonito

# h4 title row used atop every plot/render block.
@inline _block_title(title, suffix) = DOM.div(
    DOM.h4(title, style="color: #34495e; margin: 0; display: inline;"),
    DOM.span(suffix, style="color: #7f8c8d; font-size: 0.8em; font-style: italic;"),
    style="margin-bottom: 4px;",
)

_card_style = "background: white; border: 1px solid #dee2e6; border-radius: 5px; padding: 6px; width: fit-content;"

"""
    plot_block(fig, n_points_obs, show_obs, title; margin="") -> Observable DOM

A WGLMakie figure card whose visibility tracks `show_obs` and whose subtitle shows
the live point count from `n_points_obs`.
"""
function plot_block(fig, n_points_obs, show_obs, title::String; margin::String = "")
    map(show_obs) do visible
        visible || return DOM.div(style = "display: none;")
        DOM.div(
            _block_title(title, map(n -> " · $(n) pts", n_points_obs)),
            DOM.div(fig, style = _card_style),
            style = "flex: 1; min-width: 0; $(margin)",
        )
    end
end

"""
    scatter_block(fig, handle, show_obs; margin) -> Observable DOM

Per-channel hemisphere scatter card.
"""
scatter_block(fig, handle, show_obs; margin::String = "margin-right: 6px;") =
    plot_block(fig, handle.n_points, show_obs, "Scatter (per-channel)"; margin = margin)

"""
    surface_block(fig, handle, show_obs; margin) -> Observable DOM

Luminance-shaped surface card.
"""
surface_block(fig, handle, show_obs; margin::String = "margin: 0 6px;") =
    plot_block(fig, handle.n_points, show_obs, "Surface (luminance shape, RGB color)"; margin = margin)

"""
    slice_block(fig, handle, show_obs; margin) -> Observable DOM

Plane-of-incidence polar slice card (RGB + luminance cross-section).
"""
slice_block(fig, handle, show_obs; margin::String = "margin: 0 6px;") =
    plot_block(fig, handle.n_samples, show_obs, "Slice (plane of incidence)"; margin = margin)

"""
    _legend_dom(spec) -> DOM

Color-bar legend from an `error_map_legend` spec (`nothing` → empty div). The
gradient bar spans the render card's width with the three labels beneath it.
"""
function _legend_dom(spec)
    spec === nothing && return DOM.div()
    grad = "linear-gradient(to right, " * join(spec.colors, ", ") * ")"
    return DOM.div(
        DOM.div(style = "height: 10px; border-radius: 3px; background: $(grad); border: 1px solid #dee2e6;"),
        DOM.div(
            DOM.span(spec.left), DOM.span(spec.mid), DOM.span(spec.right),
            style = "display: flex; justify-content: space-between; color: #7f8c8d; font-size: 0.75em; margin-top: 1px;",
        ),
        style = "width: 400px; max-width: 100%; margin-top: 4px;",
    )
end

"""
    render_sphere_block(session, payload_obs, light_dir_obs, show_obs;
                        renderer_module, global_key, subtitle, margin="",
                        render_shape_obs=nothing, intensity_obs=nothing,
                        legend_obs=nothing, block_title="Sphere Render") -> Observable DOM

A Three.js render card driven by the universal renderer. `payload_obs` carries the
generated LUT + shader; `light_dir_obs` drives the light marker; `subtitle` may be a
`String` or an `Observable{String}`. `global_key` is the unique `window[...]` slot
that holds this sphere's renderer instance (e.g. `"_brdfMeasured"`).

When `render_shape_obs` is provided, `r.setGeometry(shape)` is called on init and
wired reactively so all render panels stay in sync with the shape selector.
`block_title` replaces the default "Sphere Render" heading and may itself be an
Observable (e.g. a reactive shape-name string).

`intensity_obs` (Observable of the absolute `uIntensity` value) overrides the
payload's display gain: applied after every `setMaterial` (initial mount AND
payload rebuilds, which would otherwise reset the gain) and live via
`updateUniforms` — a pure uniform update, no rebuild. Omit for panels whose gain
must stay fixed (the error map).

`legend_obs` (Observable of an `error_map_legend` spec or `nothing`) renders a
color-bar legend under the canvas.
"""
function render_sphere_block(session, payload_obs, light_dir_obs, show_obs;
                             renderer_module, global_key::String,
                             subtitle, margin::String = "",
                             render_shape_obs=nothing,
                             intensity_obs=nothing,
                             legend_obs=nothing,
                             block_title="Sphere Render",
                             show_wi_obs=nothing)
    # Constant fallbacks so ONE onload/onjs code path serves every panel:
    # shape "sphere" matches the renderer's initial geometry (setGeometry no-ops),
    # intensity 0 is the sentinel for "keep the payload's own uIntensity".
    shape_obs = render_shape_obs === nothing ? Observable("sphere") : render_shape_obs
    gain_obs  = intensity_obs === nothing ? Observable(0.0) : intensity_obs
    wi_vis_obs = show_wi_obs === nothing ? Observable(true) : show_wi_obs
    map(show_obs) do visible
        visible || return DOM.div(style = "display: none;")
        canvas_container = DOM.div(
            style = "width: 400px; height: 400px; background: #202020; border-radius: 5px; overflow: hidden; position: relative; width: fit-content;",
        )
        Bonito.onload(session, canvas_container, js"""
            function (el) {
                $(renderer_module).then(mod => {
                    const r = mod.createUniversalRenderer(el, $(global_key));
                    window[$(global_key)] = r;
                    const dir = $(light_dir_obs).value;
                    r.setLightDir(dir[0], dir[1]);
                    r.setLightMarkerVisible($(wi_vis_obs).value);
                    r.setGeometry($(shape_obs).value);

                    const payload = $(payload_obs).value;
                    if (payload) {
                        r.loadTextureData(payload.b64, payload.config).then(() => {
                            r.setMaterial(payload.glsl, payload.uniforms);
                            const gain = $(gain_obs).value;
                            if (gain > 0) r.updateUniforms({uIntensity: gain});
                        });
                    }
                }).catch(err => console.error('[renderer] load failed:', err));
            }
        """)
        Bonito.onjs(session, payload_obs, js"""(payload) => {
            const r = window[$(global_key)];
            if (r && payload) {
                r.loadTextureData(payload.b64, payload.config).then(() => {
                    r.setMaterial(payload.glsl, payload.uniforms);
                    const gain = $(gain_obs).value;
                    if (gain > 0) r.updateUniforms({uIntensity: gain});
                });
            }
        }""")
        Bonito.onjs(session, light_dir_obs, js"""(dir) => {
            const r = window[$(global_key)];
            if (r) r.setLightDir(dir[0], dir[1]);
        }""")
        if render_shape_obs !== nothing
            Bonito.onjs(session, render_shape_obs, js"""(shape) => {
                const r = window[$(global_key)];
                if (r) r.setGeometry(shape);
            }""")
        end
        if intensity_obs !== nothing
            Bonito.onjs(session, intensity_obs, js"""(gain) => {
                const r = window[$(global_key)];
                if (r && gain > 0) r.updateUniforms({uIntensity: gain});
            }""")
        end
        if show_wi_obs !== nothing
            Bonito.onjs(session, show_wi_obs, js"""(visible) => {
                const r = window[$(global_key)];
                if (r) r.setLightMarkerVisible(visible);
            }""")
        end
        DOM.div(
            _block_title(block_title, subtitle),
            DOM.div(canvas_container, style = _card_style),
            legend_obs === nothing ? DOM.div() : map(_legend_dom, legend_obs),
            style = "flex: 1; min-width: 0; width: fit-content; $(margin)",
        )
    end
end

"""
    viz_section(title, children...; reverse=false, title_style=...) -> (h3, body)

Returns the section heading and its flex body as a **tuple of siblings** (see the
layout note above). `reverse=true` stacks children bottom-up (`column-reverse`,
used by Measured/Fitted so the render sphere sits on top); `reverse=false` is a
plain row (used by the single-sphere Error section).
"""
function viz_section(title::String, children...; reverse::Bool = false,
                     title_style::String = "color: #2c3e50; margin: 12px 0 4px 0; font-size: 1em;")
    flex = reverse ? "display: flex; gap: 6px; flex-direction: column-reverse; width: fit-content;" :
                     "display: flex; gap: 6px; width: fit-content;"
    h3 = DOM.h3(title, style = title_style)
    body = DOM.div(children...; style = flex)
    return (h3, body)
end
