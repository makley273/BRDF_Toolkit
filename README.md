# BRDFToolkit

An interactive web platform for analysing measured BRDF data and fitting analytical
and neural BRDF models to it. Load a measured material, fit a model, and compare the
two side by side — both as scientific plots and as a rendered material under live
lighting.

Written in Julia. Bachelor thesis project by Maria Alexandra Kley.

---

## What it does

- **Loads measured BRDF data** from three databases, auto-detecting the format.
- **Fits 13 BRDF models** to that data with any of 8 optimisers, all sharing one
  fitting pipeline (sampling strategy, loss function, restarts).
- **Compares measured against fitted** in BRDF space (RMSE, RMS-CRWE, RMS-LogWE)
  and in image space (SSIM on rendered spheres).
- **Renders both** through one universal 4D lookup table, so every model — analytical
  or neural — reaches the GPU through the same shader.
- **Visualises the error** directly on the sphere with selectable error maps.

## Quick start

```bash
julia --project=.
```

```julia
] instantiate
```

Then launch the platform:

```bash
julia --project=. start_platform.jl
```

Open <http://127.0.0.1:8000>. One server hosts all three tools, linked by a
navigation bar:

| Route | Tool |
|---|---|
| `/` | BRDF analysis — load, fit, visualise, render |
| `/rusinkiewicz` | Rusinkiewicz parametrization explorer (dual view) |
| `/sampling` | Fit-sampling strategy explorer |

`--port` and `--host` override the defaults (`8000`, `0.0.0.0`).

### Programmatic use

```julia
using BRDFPlatform

data  = load_brdf("path/to/material.binary")   # format auto-detected
model = fit_brdf(LeastSquaresOptimFitter(), GGXModel, data)

brdf_rmsle(data, model)     # BRDF-space error (headline metric)
image_ssim(data, model)     # image-space agreement
```

## Supported formats

| Format | Extension | Parameterization |
|---|---|---|
| **MERL** | `.binary` | Dense 90×90×180 Rusinkiewicz half-angle grid, isotropic |
| **RGL** (EPFL) | `.bsdf` | Data-adaptive sampling warp, isotropic + anisotropic |
| **UTIA** | — | 6×48×6×48 grid over spherical angles, anisotropic |

Every loader converts to one `MeasuredBRDF`, whose `_data` field keeps the
format's own sampling knowledge internal:

```julia
struct MeasuredBRDF <: BRDFData
    _data::Any                       # format-specific, opaque
    wavelengths::Vector{Float32}     # nm
    name::String
    source_format::Symbol            # :merl, :rgl, :utia
    status::Symbol                   # :ok, :partial, :failed
    error_msg::Union{Nothing, String}
end
```

Measured databases are not redistributed with this repository; point the loader at
your own copies.

## Models

**Microfacet families** — each in a plain, specular-only, scaled-diffuse and
Fresnel-diffuse variant:

- **GGX** (Trowbridge–Reitz), plus an anisotropic form
- **Cook–Torrance** (Beckmann), plus an anisotropic form

**Others:** Ashikhmin–Shirley, Lambertian, and a **neural BRDF** (a small MLP
trained with Lux, reciprocal by construction through its input mapping).

## Optimisers

LeastSquaresOptim · Levenberg–Marquardt (LsqFit) · NonlinearSolve · L-BFGS
(`Fminbox`) · L-BFGS-B · Nelder–Mead · Particle swarm · Neural trainer (Adam).

### Fitting defaults

The defaults follow the thesis evaluation rather than convention:

| Setting | Default | Why |
|---|---|---|
| Optimiser | **LeastSquaresOptim**, 16 restarts | Reaches the converged SSIM about 5× faster than single-start L-BFGS |
| Loss | **logarithmic** | The most robust of six; uncompressed losses collapse on specular materials |
| Sampling | **cosine-weighted uniform** | Beat Rusinkiewicz warping on 294 of 312 materials by SSIM |
| Residual weighting | cosine | Down-weights noisy grazing-angle samples |

## Rendering

A model is implemented **once**, as a Julia `evaluate(model, wi, wo)`. The universal
generator samples it into a 4D lookup table over the Rusinkiewicz angles
(θh, θd, φd, φh), and a single shared shader (`src/rendering/shaders/universal.glsl`)
renders it. There is no per-model GLSL: measured data and every fitted model take the
same path to the GPU.

## Writing a plugin

A model is one self-contained file. Fitting, rendering, the UI and export are all
inherited.

```julia
# src/models/your_model.jl
struct YourModel <: BRDFModel
    # parameters
end

# The only required method. Returns a BRDFRGB; zero below the horizon.
# Keep wi/wo untyped so autodiff dual numbers flow through during fitting.
evaluate(m::YourModel, wi, wo) = ...

# Parameter packing, box bounds and initial guess, shared by every optimiser.
_fit_spec(::Type{YourModel}) = (reconstruct = p -> YourModel(...),
                                p0 = [...], lower = [...], upper = [...])

register_model!(YourModel)   # also auto-exports the type
```

Formats implement `detect_format` (confidence 0–1) and `load_brdf`, then
`register_format!`. Optimisers implement `fit_brdf(opt, ModelType, data)`, then
`register_optimizer!`.

User plugins are picked up from outside the repository:

```
~/.brdf_platform/formats/
~/.brdf_platform/models/
~/.brdf_platform/optimizers/
```

```julia
BRDFPlatform.load_user_plugins()
```

## Conventions

- All direction vectors are normalised; the surface normal is +z.
- `wi` and `wo` both point **away** from the surface.
- Angles in radians.
- RGB channels approximate 630 / 532 / 465 nm.

## Tests

```bash
julia --project=. test/runtests.jl
```

The suite is data-file free and runs on a fresh clone. The format loader tests in
`test/` (MERL, RGL, UTIA) are excluded from it because they need measured databases;
run them directly once you have pointed them at your own copies.

## Project layout

```
src/
├── BRDFPlatform.jl      # module, plugin discovery
├── types.jl             # abstract types, MeasuredBRDF, registries
├── analysis.jl          # error metrics (RMSE, CRWE, LogWE, SSIM)
├── benchmarks.jl        # thesis statistics tables
├── formats/             # merl.jl, rgl.jl, utia.jl
├── models/              # 13 model plugins + shared fit specs
├── optimizers/          # 8 fitting backends + shared pipeline
├── errormaps/           # selectable error visualisations
├── rendering/           # 4D-LUT generator, shared shader, CPU render for SSIM
├── visualization/       # WGLMakie scientific plots
└── app/                 # Bonito web UI (analysis, Rusinkiewicz, sampling)
```

## Stack

Julia · Bonito.jl (web) · WGLMakie (scientific plots) · Three.js (material
rendering) · Lux.jl (neural models) · Optim / LsqFit / LeastSquaresOptim /
NonlinearSolve (fitting).

## Author

Maria Alexandra Kley — Bachelor thesis in Computer Graphics.
