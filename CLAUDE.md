# CLAUDE.md — AI Assistant Guide

**Project:** BRDFToolkit (Julia module `BRDFPlatform`)
**Type:** Bachelor thesis, Computer Graphics — Maria Alexandra Kley
**Status:** v1.0.0 released. The implementation is complete; work now is
maintenance, polish and thesis support, not new phases.

`README.md` is the user-facing overview. This file is for working *on* the code.

---

## What the platform does

Loads measured BRDF data, fits analytical and neural BRDF models to it, and
compares measured against fitted — as metrics, as scientific plots, and as a
rendered material. Everything is a plugin: 3 formats, 13 models, 8 optimisers.

| Plugin kind | Registered today |
|---|---|
| Formats | `MERLFormat` (`.binary`), `RGLFormat` (`.bsdf`), `UTIAFormat` |
| Models | Lambertian, Ashikhmin–Shirley, GGX ×5, Cook–Torrance ×5, NeuralBRDF |
| Optimisers | LeastSquaresOptimFitter, LevenbergMarquardt, NonlinearSolveFitter, LBFGS, LBFGSB, NelderMeadOpt, ParticleSwarmOpt, NeuralBRDFTrainer |

The GGX and Cook–Torrance families each come as plain, `Specular`,
`ScaledDiffuse`, `FresnelDiffuse` and `Anisotropic` variants.

**BiRD is not supported.** Its loader and exporter were removed; do not
reintroduce BiRD references in code or docs.

---

## Technology stack — fixed, do not change

- **Julia** — all computation, fitting, scientific plots
- **Bonito.jl** — web framework (not Genie.jl)
- **WGLMakie** — interactive scientific plots; **CairoMakie** — static figure export
- **Three.js** — material rendering, embedded in Bonito (no standalone JS framework)
- **GLSL** — exactly one shared shader, `universal.glsl`
- **Lux.jl** — neural BRDF model
- **Optim / LsqFit / LeastSquaresOptim / NonlinearSolve** — fitting backends

Not allowed: WASM for BRDF computation, a standalone JS framework, other web
frameworks.

---

## Architecture

### Plugin discovery

`__init__` calls `load_core_plugins()` (every `.jl` in `src/formats/`,
`src/models/`, `src/optimizers/`, then `visualization/plots.jl`) and
`load_user_plugins()` (same layout under `~/.brdf_platform/`). Files load in
`readdir` order, so shared vocabulary files start with an underscore
(`_fit_spec.jl`, `_microfacet_utils.jl`, `_fit_common.jl`) to load first.

`register_format!` / `register_model!` / `register_optimizer!` add the type to its
registry under `Symbol(lowercase(typename))` **and export it** — plugin files need
no `export` line.

### Data flow

```
file → detect_format (highest confidence) → MeasuredBRDF
     → fit_brdf(optimizer, ModelType, data) → BRDFModel
     → metrics:   brdf_rmse / brdf_rms_crwe / brdf_rmsle / image_ssim
     → plots:     WGLMakie (scatter, surface, polar slice)
     → rendering: evaluate → generate_lut (4D) → universal.glsl
```

### Single implementation per model

A model is implemented **once**, as a Julia `evaluate`. It drives fitting, error
metrics and rendering alike. There is no per-model GLSL, no `glsl_shader` method,
and no Julia-vs-GLSL validation — that dual layer was removed.

### Rendering

`generate_lut` (`src/rendering/universal_texture.jl`) samples any `MeasuredBRDF`
into a 4D table over the Rusinkiewicz angles (θh, θd, φd, φh).
`universal.glsl` + `src/rendering/js/renderer.js` render it. Measured data and
every fitted model take this same path. The LUT layout must stay in lock-step
across those three files.

- φd is stored over the **full circle** `[0, 2π]`. Never fold it — folding
  produces a star/spoke artifact.
- Unreachable (θh, θd) cells are filled from the nearest realizable pair, not
  from an impossible geometry.
- `renderer.bundled.js` is Bonito's build output from `ES6Module`; it is
  gitignored on purpose.
- `src/rendering/cpu_render.jl` is a CPU twin of the Three.js renderer (same
  camera, light, Reinhard tone map, no gamma) used by `image_ssim`.

### Error maps

`src/errormaps/error_maps.jl` defines the abstract `ErrorMap` with
`SignedLogDifference` (default), `RelativeErrorRGB` and `SMAPEHeat`, listed in
`AVAILABLE_ERROR_MAPS`. `error_to_measured` bakes an error map into a
`MeasuredBRDF` so it renders through the same LUT path (unlit).

---

## Project structure

```
src/
├── BRDFPlatform.jl          # module, plugin discovery, exports
├── types.jl                 # abstract types, BRDFRGB, MeasuredBRDF, registries
├── analysis.jl              # metrics: RMSE, RMS-CRWE, RMS-LogWE, SSIM, reciprocity
├── benchmarks.jl            # thesis statistics tables (CSV/MD/LaTeX)
├── formats/                 # merl.jl, rgl.jl, utia.jl
├── models/                  # _fit_spec.jl, _microfacet_utils.jl + 13 models
├── optimizers/              # _fit_common.jl (shared pipeline) + 8 backends
├── errormaps/error_maps.jl
├── rendering/               # universal_texture.jl, cpu_render.jl,
│                            # shaders/universal.glsl, js/renderer.js
├── visualization/plots.jl   # WGLMakie plot handles
└── app/
    ├── hub.jl                        # run_platform: one server, nav bar, routes
    ├── interactive_app.jl            # "/"  analysis UI
    ├── rusinkiewicz_vizualization.jl # "/rusinkiewicz"  dual-view explorer
    ├── fit_sampling_app.jl           # "/sampling"  sampling-strategy explorer
    ├── viz_blocks.jl                 # reusable DOM blocks (plot cards, spheres)
    └── styles/app.css
test/        # runtests.jl (data-free) + standalone loader/visual tests
tools/       # thesis experiment scripts (sweeps, stats, RQ5 review apps)
start_platform.jl            # the only launcher
```

---

## Plugin interfaces

### Format

```julia
struct YourFormat <: BRDFData end
detect_format(::Type{YourFormat}, path::String)::Float64   # confidence 0–1
load_brdf(::Type{YourFormat}, path::String)::MeasuredBRDF  # wraps YourData
_brdf_value(d::YourData, wavelengths, wi, wo)              # lookup
n_samples_str(d::YourData)                                 # UI display
register_format!(YourFormat)
```

Sampling knowledge stays inside the format's own data struct (`_data` of the
`MeasuredBRDF` is opaque). There is no separate sampling abstraction.

### Model

```julia
struct YourModel <: BRDFModel ... end
evaluate(m::YourModel, wi, wo)          # -> BRDFRGB{T}; the only required method
_fit_spec(::Type{YourModel}) =          # NamedTuple, lives in the model file
    (reconstruct = p -> YourModel(...), p0 = [...], lower = [...], upper = [...])
register_model!(YourModel)
```

- Leave `wi`/`wo` **untyped** and let `evaluate` return `BRDFRGB{T}` with `T`
  inferred — ForwardDiff dual numbers must flow through for the Jacobian.
- Return zero below the horizon (`wi[3] <= 0 || wo[3] <= 0`).
- Reuse the bound constants and microfacet factories in `_fit_spec.jl`
  (roughness `[0.02, 1]`, F0/albedo/kd `[0, 1]`) instead of new literals.
- `display_parameters` is optional; the generic fallback reflects struct fields.
- `lambertian.jl` is the minimal reference.

### Optimiser

```julia
struct YourOptimizer <: BRDFOptimizer ... end
fit_brdf(opt::YourOptimizer, ::Type{M}, data::MeasuredBRDF)::M where {M <: BRDFModel}
optimizer_options(::Type{YourOptimizer}) = OptimizerOption[...]   # drives the UI
register_optimizer!(YourOptimizer)
```

Build on `src/optimizers/_fit_common.jl` rather than re-implementing it:
`_sample_directions(sampling, n)` (valid symbols in `VALID_SAMPLINGS`),
`_residual_function(loss)`, `_fit_targets` / `_fit_predict` (cosine weighting),
and `_generate_starts` (multi-start). The UI builds its option controls from
`optimizer_options`, so a struct default and its `OptimizerOption` default must
match.

---

## Fitting and metrics conventions

### Defaults (from the thesis evaluation — keep them)

| Setting | Default |
|---|---|
| App default optimiser | `LeastSquaresOptimFitter` |
| Restarts | 16 for LeastSquaresOptim, 1 for the others |
| Loss | `:log` |
| Sampling | `:uniform_cosine` for all analytical optimisers; the neural trainer keeps `:rusinkiewicz` |
| Cosine residual weighting | on |
| Direction pairs | 1024 (neural trainer: 8192) |

### Loss functions

Every loss is a **pseudo-residual** `R` whose square is the intended penalty,
because every solver squares and sums `R`. That is what lets residual-vector
solvers (LM family) and scalar solvers (L-BFGS, Nelder–Mead, swarm) share one
interface. Robust losses (Huber, log-cosh) have branches that keep derivatives
finite for autodiff.

### Metrics

- **RMS-LogWE (`brdf_rmsle`) is the headline BRDF-space metric**, not RMS-CRWE.
  Report it in units of 1e-3.
- All three BRDF-space metrics are cosine-weighted before their transform.
- `image_ssim` returns `(mean, min, per_light)`. Reinhard tone mapping flattens
  specular error, so SSIM can read ~0.99 while a highlight is badly wrong —
  **always pair SSIM with RMS-LogWE**.
- Fitted parameters are not portable across tools (e.g. `dj_brdf` fits roughness
  against a Fresnel spline, not Schlick F0). Compare fits by metric, never by
  parameter value.

---

## Coordinate conventions

- All direction vectors normalized; surface normal is +z.
- `wi` and `wo` both point **away** from the surface.
- Angles in radians.
- RGB channels approximate 630 / 532 / 465 nm.
- Rusinkiewicz: θh ∈ [0, π/2], φh ∈ [0, 2π), θd ∈ [0, π/2], φd ∈ [0, 2π].
  MERL tables are isotropic (φh = 0) and index θh with a square-root warp.

---

## Web apps

- `julia --project=. start_platform.jl [--port N] [--host H]` is the **only**
  launcher. It calls `run_platform` (default port 8000).
- Routes: `/` analysis, `/rusinkiewicz`, `/sampling`. Standalone runners also
  exist (`run_interactive_app`, `run_rusinkiewicz_app` 8060, `run_sampling_app`
  8050).
- Servers bind `0.0.0.0` and use an origin-relative `proxy_url` via
  `resolve_proxy_url`, so LAN/VPN access works.
- View syncing: `_sync_axis3_views!(axes...; enabled)` links `Axis3` plots
  (azimuth, elevation, targetlimits); `_sync_polar_views!` links `PolarAxis`
  plots through their `target_*` observables. In the analysis app one
  `channel_vis` tuple drives the R/G/B/Lum toggles of measured *and* fitted
  plots.
- Never make per-frame Julia↔JS calls. Generate the LUT once per data/fit,
  upload it, then change uniforms only.

---

## Testing

```bash
julia --project=. test/runtests.jl
```

`runtests.jl` is **data-file free** and must pass on a fresh clone. It covers the
module/registry basics plus `test_ashikhmin_shirley`, `test_neural_brdf`,
`test_image_metrics`, `test_error_maps`, `test_lut_sampling` and
`test_benchmarks` (on a synthetic BRDF via `_model_to_measured`).

Measured databases live in gitignored `test/data/{merl,utia,RGL}/`. The loader
tests that need them (`test_merl_loader.jl`, `test_rgl_loader.jl`,
`test_utia_loader.jl`) and the visual/server smoke scripts are run directly, not
from `runtests.jl`. `test/merl.cpp` is the MERL reference reader whose output
the MERL test transcribes.

For a new model, validate `evaluate` against analytic identities and physical
bounds (reciprocity, non-negativity, energy sanity) — see
`test_ashikhmin_shirley.jl`.

---

## Development guidelines

**Prioritise:** correctness over performance, clarity over cleverness, research
goals over engineering perfection, working features over perfect architecture.

**Do**
- Keep the flat directory structure and explicit plugin registration.
- Implement one Julia `evaluate` per model.
- Degrade gracefully: failed loads return `status = :failed` with a message;
  surface errors in the UI, never only in the console.
- Document coordinate systems and units.

**Don't**
- Create deep module hierarchies or macro-based plugin DSLs.
- Write per-model GLSL.
- Change the stack.
- Over-engineer for flexibility beyond what the platform needs.
- **Create extra documents when fixing a problem.** Fix only what's broken. No new
  markdown files, summaries or write-ups unless explicitly asked.

**Gotchas**
- Julia can't redefine a struct in a running session: changing a struct's
  fields needs a restart.
- `update_slice_plot!` and the hemisphere plots recompute on every observable
  change they subscribe to; add new triggers through `_wire_plots!` rather than
  ad-hoc `on(...)` handlers so measured and fitted panels stay in step.
