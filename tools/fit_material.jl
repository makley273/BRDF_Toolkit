#!/usr/bin/env julia
"""
tools/fit_material.jl — fit BRDF models to ONE material across a sweep of fitting
factors, and export the results as a thesis-ready bundle (renders + tables).

This is the batch driver behind two Results-chapter studies
(`.thesis/latex/content/results.tex`):

  * "Model Expressiveness Across Material Classes" (`sec:results-expressiveness`)
    — fit every analytical model (and, optionally, the neural model) to a
    material and score each in BRDF space (RMS-LogWE, RMS-CRWE, RMSE) and image space
    (SSIM mean + worst-case light).

  * "Fitting Strategies" (`sec:results-fitting`) — one-factor-at-a-time (or full
    grid) comparison of optimizer, loss, sampling, sample count, and multi-start
    around a default configuration.

Any of `--models --optimizer --loss --sampling --samples --starts` accepts a
COMMA-SEPARATED LIST. The tool fits the full Cartesian product of the lists (one
"run" per combination), then builds a table whose columns are exactly the factors
you varied (constant factors go in the caption). Passing single values reproduces
the original per-model behaviour.

Renders come from the CPU sphere renderer (`render_brdf_sphere`), the exact Julia
twin of the browser Three.js path — so they carry NO light-direction indicator
(that arrow lives only in the interactive WebGL scene). Each render is cropped
by `RENDER_CROP_FRACTION` of empty margin on every side and saved as a JPEG at
`RENDER_JPEG_QUALITY`% quality.

Usage
-----
    julia --project=. tools/fit_material.jl <material> [options]

`<material>` is a path to a BRDF file, OR a path to a folder — in which case
every file directly inside it is tried in turn, exactly as if it had been passed
as `<material>` on its own. The path you give is the only place looked at: no
search directories, no extension guessing. Whether a file is a valid measured
BRDF is decided by `load_brdf`, so anything that fails to load (or to fit) is
skipped with a warning and the rest continue.

Options (defaults shown). Any of the *swept* options takes a comma list; the
factor axes marked [all] also accept the literal "all" to sweep every value:
    --models      all                 "all" or e.g. "CookTorrance,GGX,Lambertian"   [swept, all]
    --optimizer   lm                  lm|lbfgs|lbfgsb|neldermead|particleswarm|nls|lsof  [swept, all]
    --loss        cuberoot            log|linear|cuberoot|ngan|huber|logcosh         [swept, all]
    --sampling    rusinkiewicz        rusinkiewicz|uniform_half|mixed|uniform|...    [swept, all]
    --samples     2048                (wi,wo) pairs used for fitting                 [swept]
    --starts      1                   multi-start count (analytical optimizers)      [swept]
    --cosine-weight true              cosine (cos θi) residual weighting; true|false|all  [swept]
    --neural                          also fit the neural MLP model (loads Lux/CUDA)
    --neural-hidden 21,21             neural arch(s); ';'-separate archs to sweep them, e.g. "21,21;32,32;64,64"
    --neural-anisotropic false        6-input aniso feature mapping; true|false|all (swept, neural only)
    --neural-epochs 150               neural training epochs
    --neural-hidden-act relu          hidden activation (relu|gelu|tanh|softplus)
    --neural-out-act softplus         output activation (softplus|exp|square|abs|relu)
    --neural-optimizer adam           adam|adamw
    --neural-lr 0.001                 Adam learning rate
    --neural-weight-decay 0.0001      AdamW weight decay
    --neural-batch 512                minibatch size
    --neural-seed 1                   RNG seed
    --neural-gpu                      train on CUDA GPU if available
    --resolution  400                 render resolution (px, square)
    --lights      ssim                "ssim" (3 SSIM lights) | "front" | "th:ph,..." in degrees
    --render      all                 all | best (best run per model) | none
    --errormaps                       also render a relative-error sphere per rendered run
    --stats                           also write thesis-statistics tables (storage / eval timing / LUT) via src/benchmarks.jl
    --metric-samples 4096             direction pairs for the BRDF-space metrics
    --out         <auto>              output dir (default exports/fits/<material>_<timestamp>,
                                       or exports/fits/<folder>_<timestamp> for a folder input)

Output — ONE material
---------------------
    <out>/renders/<run-slug>__th<θ>_ph<φ>.jpg        (run-slug encodes the varied factors)
    <out>/renders/<run-slug>__err__th<θ>_ph<φ>.jpg   (with --errormaps)
    <out>/renders/measured__th<θ>_ph<φ>.jpg
    <out>/metrics.csv   (full per-run record)   <out>/metrics.md   <out>/metrics.tex
    <out>/stats/{storage,model_storage,eval_timing,lut}.{csv,md,tex}   (with --stats)

Output — SEVERAL materials (folder input)
----------------------------------------
Each material keeps its own bundle exactly as above, in its own subdirectory,
and the results are additionally MERGED into one set of tables at the top:

    <out>/<material>/...                             (the per-material bundle)
    <out>/metrics_all.{csv,md,tex}                   every run of every material,
                                                     with a leading Material column
    <out>/stats/{storage,model_storage,eval_timing,lut}.{csv,md,tex}   (with --stats)

The merged stats reuse the per-material numbers, so nothing is benchmarked twice.
"""

using BRDFPlatform
using JpegTurbo
using Colors
using Printf
using Dates

# ============================================================================
# JPEG export (cropped, pixel-exact — verified against a marker image)
# ============================================================================

# Renders have empty margin around the sphere; crop it before export. The margin
# is a fraction of the render size rather than a pixel count, since the empty
# border scales with `--resolution`: 0.175 is the 70 px the default 400 px render
# was calibrated with, so that case is unchanged and higher-resolution runs get
# the same framing instead of a proportionally larger border.
const RENDER_CROP_FRACTION = 0.175
const RENDER_JPEG_QUALITY = 80

# Drop `margin` px from each side. Falls back to the uncropped image if it's
# too small for the requested margin (e.g. a low --resolution run).
function crop_margin(img::AbstractMatrix, margin::Int)
    h, w = size(img)
    (margin <= 0 || h <= 2margin || w <= 2margin) && return img
    return img[margin+1:h-margin, margin+1:w-margin]
end

"""
    save_render(path, img; margin=nothing, quality=RENDER_JPEG_QUALITY)

Crop empty space from each side of a rendered sphere image, then write it as a
JPEG at `quality` (0-100). `margin` defaults to `RENDER_CROP_FRACTION` of the
image size, so the framing is the same at any `--resolution`; pass an explicit
pixel count to override. `img` is a `Matrix{RGB}` in the `[row, col]`
top-left-origin convention `render_brdf_sphere` returns, which is exactly what
`JpegTurbo.jpeg_encode` expects — no rotation needed.
"""
function save_render(path::AbstractString, img::AbstractMatrix{<:Colorant};
                     margin::Union{Nothing, Int} = nothing,
                     quality::Int = RENDER_JPEG_QUALITY)
    m = margin === nothing ?
        round(Int, RENDER_CROP_FRACTION * minimum(size(img))) : margin
    cropped = crop_margin(img, m)
    jpeg_encode(path, cropped; quality = quality)
    return path
end

# ============================================================================
# Model resolution
# ============================================================================

# Preferred display / fit order (increasing complexity). Only those actually
# registered are used; anything registered but unlisted is appended alphabetically.
const _MODEL_ORDER = [
    "lambertianmodel",
    "cooktorrancespecularmodel", "cooktorrancemodel",
    "cooktorrancescaleddiffusemodel", "cooktorrancefresneldiffusemodel",
    "ggxspecularmodel", "ggxmodel",
    "ggxscaleddiffusemodel", "ggxfresneldiffusemodel",
    "ashikhminshirleymodel",
    "cooktorranceanisotropicmodel", "ggxanisotropicmodel",
]

_is_neural(name::AbstractString) = occursin("neural", lowercase(name))

"""
    analytical_model_types() -> Vector{Type}

Every registered `BRDFModel` except the neural one, in `_MODEL_ORDER` (registered
extras appended alphabetically).
"""
function analytical_model_types()
    reg = MODEL_REGISTRY                       # Dict{Symbol, Type}
    keys_lc = Dict(lowercase(String(k)) => k for k in keys(reg))
    ordered = String[]
    for nm in _MODEL_ORDER
        haskey(keys_lc, nm) && push!(ordered, nm)
    end
    for nm in sort(collect(keys(keys_lc)))
        (nm in ordered || _is_neural(nm)) && continue
        push!(ordered, nm)
    end
    return [reg[keys_lc[nm]] for nm in ordered]
end

"""
    resolve_models(spec) -> Vector{Type}

`spec == "all"` → every analytical model. Otherwise a comma list of model names
(case-insensitive, trailing "Model" optional, e.g. "cook_torrance", "GGX").
"""
function resolve_models(spec::AbstractString)
    lowercase(strip(spec)) == "all" && return analytical_model_types()
    reg = MODEL_REGISTRY
    alias = Dict{String, Type}()
    for (k, T) in reg
        base = lowercase(String(k))
        alias[base] = T
        endswith(base, "model") && (alias[base[1:end-5]] = T)
    end
    norm(s) = replace(lowercase(strip(s)), "_" => "", "-" => "", " " => "")
    alias2 = Dict(norm(k) => v for (k, v) in alias)
    out = Type[]
    for tok in split(spec, ',')
        isempty(strip(tok)) && continue
        key = norm(tok)
        T = get(alias2, key, nothing)
        T === nothing && error("unknown model \"$tok\". Registered: " *
                               join(sort(String.(collect(keys(reg)))), ", "))
        push!(out, T)
    end
    return out
end

# ============================================================================
# Optimizer resolution
# ============================================================================

# User-facing aliases → registry key (lowercased struct name).
const _OPT_ALIAS = Dict(
    "lm"             => "levenbergmarquardt",
    "levenbergmarquardt" => "levenbergmarquardt",
    "lbfgsb"         => "lbfgsb",
    "lbfgs"          => "lbfgs",
    "fminbox"        => "lbfgs",
    "neldermead"     => "neldermeadopt",
    "nm"             => "neldermeadopt",
    "particleswarm"  => "particleswarmopt",
    "pso"            => "particleswarmopt",
    "nls"            => "nonlinearsolvefitter",
    "lsof"           => "leastsquaresoptimfitter",
)

# Registry key (lowercased struct name) → short display/CLI name.
const _OPT_CANON = Dict(
    "levenbergmarquardt"      => "lm",
    "lbfgs"                   => "lbfgs",
    "lbfgsb"                  => "lbfgsb",
    "neldermeadopt"           => "neldermead",
    "particleswarmopt"        => "particleswarm",
    "nonlinearsolvefitter"    => "nls",
    "leastsquaresoptimfitter" => "lsof",
)
# Preferred order when expanding `--optimizer all`.
const _OPT_ORDER = ["lm", "lbfgs", "lbfgsb", "nls", "lsof", "neldermead", "particleswarm"]

"""
    resolve_optimizers(spec) -> Vector{String}

`spec == "all"` → every registered analytical optimizer (neural trainer
excluded), in `_OPT_ORDER`. Otherwise a comma list of aliases/registry names.
Each is canonicalised to its short display name and de-duplicated.
"""
function resolve_optimizers(spec::AbstractString)
    isall = lowercase(strip(spec)) == "all"
    toks = isall ?
        [String(k) for k in keys(OPTIMIZER_REGISTRY) if !occursin("neural", lowercase(String(k)))] :
        _splitlist(spec)
    out = String[]
    for t in toks
        key = replace(lowercase(strip(String(t))), "_" => "", "-" => "")
        regkey = get(_OPT_ALIAS, key, key)
        disp = get(_OPT_CANON, regkey, lowercase(strip(String(t))))
        disp in out || push!(out, disp)
    end
    if isall
        rank(x) = (i = findfirst(==(x), _OPT_ORDER); i === nothing ? length(_OPT_ORDER) + 1 : i)
        sort!(out; by = rank)
    end
    return out
end

"""
    build_optimizer(name; loss, sampling, samples, starts, cosine) -> BRDFOptimizer

Construct any registered analytical optimizer from the shared knobs. Every
analytical optimizer accepts `n_samples`, `loss`, `sampling`, `n_starts`,
`cosine_weight`.
"""
function build_optimizer(name::AbstractString; loss, sampling, samples, starts, cosine)
    key = get(_OPT_ALIAS, replace(lowercase(strip(name)), "_" => "", "-" => ""), lowercase(strip(name)))
    sym = Symbol(key)
    haskey(OPTIMIZER_REGISTRY, sym) ||
        error("unknown optimizer \"$name\". Known aliases: " *
              join(sort(collect(keys(_OPT_ALIAS))), ", ") *
              ". Registered: " * join(sort(String.(collect(keys(OPTIMIZER_REGISTRY)))), ", "))
    T = OPTIMIZER_REGISTRY[sym]
    return T(; loss = loss, sampling = sampling, n_samples = samples, n_starts = starts,
              cosine_weight = cosine)
end

# All valid analytical losses (mirrors each optimizer's constructor guard).
const _ALL_LOSSES = [:log, :linear, :cuberoot, :ngan, :huber, :logcosh]

resolve_losses(spec::AbstractString) =
    lowercase(strip(spec)) == "all" ? copy(_ALL_LOSSES) : Symbol.(_splitlist(spec))

resolve_samplings(spec::AbstractString) =
    lowercase(strip(spec)) == "all" ? collect(BRDFPlatform.VALID_SAMPLINGS) : Symbol.(_splitlist(spec))

# Parse a bool sweep list: "true"/"false"/"all" (→ both) or a comma list.
function resolve_bools(spec::AbstractString)
    lowercase(strip(spec)) == "all" && return [true, false]
    return [lowercase(strip(x)) in ("true", "1", "yes", "on") for x in _splitlist(spec)]
end

# ============================================================================
# Light resolution
# ============================================================================

function resolve_lights(spec::AbstractString)
    s = lowercase(strip(spec))
    s == "ssim"  && return collect(SSIM_LIGHT_DIRS)
    s == "front" && return [(Float32(π/8), 0.0f0)]
    dirs = Tuple{Float32,Float32}[]
    for pair in split(spec, ',')
        isempty(strip(pair)) && continue
        th, ph = split(pair, ':')
        push!(dirs, (Float32(deg2rad(parse(Float64, th))),
                     Float32(deg2rad(parse(Float64, ph)))))
    end
    return dirs
end

_light_tag(th, ph) = @sprintf("th%03d_ph%03d", round(Int, rad2deg(th)), round(Int, rad2deg(ph)))

# ============================================================================
# Run configuration + factor machinery
# ============================================================================

"""
    RunConfig

One point in the sweep: a model plus the fitting factors it is fit under.
`optimizer == "neural"` marks a neural-trainer run (`hidden`/`epochs` used,
`starts` ignored); analytical runs leave `hidden === nothing`.
"""
struct RunConfig
    model::Type
    optimizer::String
    loss::Symbol
    sampling::Symbol
    samples::Int
    starts::Int
    cosine::Bool
    hidden::Union{Nothing, Vector{Int}}   # neural only (nothing ⇒ analytical run)
    epochs::Int                           # neural only
    anisotropic::Bool                     # neural only: 6-input aniso feature mapping
    neural_kwargs::Vector{Pair{Symbol, Any}}  # extra fixed NeuralBRDFTrainer knobs
end

# Swept factors, in table-column order. `model` is handled separately (always shown).
const FACTORS = [:optimizer, :loss, :sampling, :samples, :starts, :cosine, :arch, :aniso]
const FACTOR_LABEL = Dict(:optimizer => "Optimizer", :loss => "Loss",
                          :sampling => "Sampling", :samples => "Samples",
                          :starts => "Starts", :cosine => "CosWeight",
                          :arch => "Arch", :aniso => "Aniso")
const FACTOR_ABBR = Dict(:optimizer => "opt", :loss => "loss", :sampling => "samp",
                         :samples => "n", :starts => "s", :cosine => "cos",
                         :arch => "h", :aniso => "aniso")

_is_neural_run(run::RunConfig) = run.hidden !== nothing

function factor_str(run::RunConfig, key::Symbol)
    key === :optimizer && return run.optimizer
    key === :loss      && return String(run.loss)
    key === :sampling  && return String(run.sampling)
    key === :samples   && return string(run.samples)
    key === :starts    && return run.optimizer == "neural" ? "--" : string(run.starts)
    key === :cosine    && return run.cosine ? "true" : "false"
    key === :arch      && return _is_neural_run(run) ? join(run.hidden, "x") : "--"
    key === :aniso     && return _is_neural_run(run) ? (run.anisotropic ? "true" : "false") : "--"
    return "?"
end

"""
    build_runs(...) -> Vector{RunConfig}

Cartesian product of the analytical factor lists, then (if `neural`) the neural
runs over archs × loss × sampling × samples.
"""
function build_runs(models, optimizers, losses, samplings, sampleslist, startslist,
                    cosines, neural::Bool, hiddens, epochs::Int, anisos, neural_kwargs)
    runs = RunConfig[]
    for m in models, o in optimizers, l in losses, s in samplings,
        n in sampleslist, st in startslist, cw in cosines
        push!(runs, RunConfig(m, o, l, s, n, st, cw, nothing, epochs, false, Pair{Symbol,Any}[]))
    end
    if neural
        for h in hiddens, an in anisos, l in losses, s in samplings, n in sampleslist, cw in cosines
            push!(runs, RunConfig(NeuralBRDFModel, "neural", l, s, n, 1, cw, h, epochs,
                                  an, neural_kwargs))
        end
    end
    return runs
end

# ============================================================================
# Parameter count
# ============================================================================

function n_params(model::BRDFModel, T::Type)
    if model isa NeuralBRDFModel
        return sum(length, model.Ws; init = 0) + sum(length, model.bs; init = 0)
    end
    try
        return length(BRDFPlatform._fit_spec(T).p0)
    catch
        return -1
    end
end

# ============================================================================
# Fitting one run
# ============================================================================

struct FitResult
    run::RunConfig
    name::String
    model::Union{BRDFModel, Nothing}
    n_params::Int
    rms_crwe::Float64
    rmse::Float64
    rmsle::Float64
    ssim_mean::Float64
    ssim_min::Float64
    seconds::Float64
    ok::Bool
    err::String
end

function make_optimizer(run::RunConfig)
    if run.optimizer == "neural"
        return NeuralBRDFTrainer(; hidden = run.hidden, epochs = run.epochs,
                                   anisotropic = run.anisotropic,
                                   loss = run.loss, sampling = run.sampling,
                                   n_samples = run.samples, cosine_weight = run.cosine,
                                   run.neural_kwargs...)
    end
    return build_optimizer(run.optimizer; loss = run.loss, sampling = run.sampling,
                           samples = run.samples, starts = run.starts, cosine = run.cosine)
end

function fit_one(run::RunConfig, data::MeasuredBRDF, metric_n::Int)
    name = String(nameof(run.model))
    try
        opt = make_optimizer(run)
        model = nothing
        secs = @elapsed (model = fit_brdf(opt, run.model, data))
        crwe = brdf_rms_crwe(data, model; n = metric_n)
        rmse = brdf_rmse(data, model; n = metric_n)
        rmsle = brdf_rmsle(data, model; n = metric_n)
        ss   = image_ssim(data, model)
        return FitResult(run, name, model, n_params(model, run.model),
                         crwe, rmse, rmsle, ss.mean, ss.min, secs, true, "")
    catch e
        @warn "fit failed for $name" exception = e
        return FitResult(run, name, nothing, -1, NaN, NaN, NaN, NaN, NaN, NaN,
                         false, sprint(showerror, e))
    end
end

# ============================================================================
# Varied-factor detection + run slugs
# ============================================================================

# A factor "varies" iff it takes >1 distinct real value (the "--" sentinel, used
# where a factor doesn't apply to a run, is ignored).
function varied_factors(results::Vector{FitResult})
    varied = Symbol[]
    for key in FACTORS
        vals = unique(filter(!=("--"), [factor_str(r.run, key) for r in results]))
        length(vals) > 1 && push!(varied, key)
    end
    return varied
end

function fixed_factors(results::Vector{FitResult}, varied::Vector{Symbol})
    fixed = Tuple{String, String}[]
    for key in FACTORS
        key in varied && continue
        vals = unique(filter(!=("--"), [factor_str(r.run, key) for r in results]))
        length(vals) == 1 && push!(fixed, (FACTOR_LABEL[key], vals[1]))
    end
    return fixed
end

function run_slug(run::RunConfig, varied::Vector{Symbol})
    parts = [String(nameof(run.model))]
    for key in varied
        v = factor_str(run, key)
        v == "--" && continue
        push!(parts, "$(FACTOR_ABBR[key])-$(v)")
    end
    return join(parts, "__")
end

# Assign a unique slug per result (append an index on the rare collision).
function assign_slugs(results::Vector{FitResult}, varied::Vector{Symbol})
    seen = Dict{String, Int}()
    slugs = Vector{String}(undef, length(results))
    for (i, r) in enumerate(results)
        base = run_slug(r.run, varied)
        n = get(seen, base, 0) + 1
        seen[base] = n
        slugs[i] = n == 1 ? base : "$(base)__$(n)"
    end
    return slugs
end

# ============================================================================
# Rendering
# ============================================================================

function render_runs(results::Vector{FitResult}, slugs::Vector{String},
                     data::MeasuredBRDF, lights, renders_dir::String;
                     resolution::Int, errormaps::Bool, mode::String)
    mkpath(renders_dir)
    # Measured reference (per light), once.
    for (th, ph) in lights
        img, _ = render_brdf_sphere(data, th, ph; resolution = resolution)
        save_render(joinpath(renders_dir, "measured__$(_light_tag(th, ph)).jpg"), img)
    end
    mode == "none" && return 1  # measured only

    # Which runs to render?
    idxs = collect(eachindex(results))
    filter!(i -> results[i].ok, idxs)
    if mode == "best"
        # keep the lowest-RMS-LogWE run per model name
        best = Dict{String, Int}()
        for i in idxs
            j = get(best, results[i].name, 0)
            if j == 0 || results[i].rmsle < results[j].rmsle
                best[results[i].name] = i
            end
        end
        idxs = sort(collect(values(best)))
    end

    em = RelativeErrorRGB()
    count = length(lights)  # measured pngs
    for i in idxs
        r = results[i]
        for (th, ph) in lights
            img, _ = render_brdf_sphere(r.model, th, ph; resolution = resolution)
            save_render(joinpath(renders_dir, "$(slugs[i])__$(_light_tag(th, ph)).jpg"), img)
            count += 1
            if errormaps
                errbrdf = error_to_measured(data, r.model, em, data.wavelengths, "$(r.name) error")
                eimg, _ = render_brdf_sphere(errbrdf, th, ph; resolution = resolution)
                save_render(joinpath(renders_dir, "$(slugs[i])__err__$(_light_tag(th, ph)).jpg"), eimg)
                count += 1
            end
        end
    end
    return count
end

# ============================================================================
# Thesis statistics (storage / timing / LUT) via src/benchmarks.jl
# ============================================================================

"""
    best_ok_results(results) -> Vector{FitResult}

The lowest-RMS-LogWE successful run per model type. Parameter count, struct size,
and per-call evaluation time depend only on the model *type*, not on the fit, so
the stats benchmarks run once per model on its best fit rather than per run.
"""
function best_ok_results(results::Vector{FitResult})
    best = Dict{String, FitResult}()
    for r in results
        r.ok || continue
        b = get(best, r.name, nothing)
        if b === nothing || r.rmsle < b.rmsle
            best[r.name] = r
        end
    end
    return [best[k] for k in sort(collect(keys(best)))]
end

"""
    material_stats(data, material_path, best_results; n_dirs) -> NamedTuple of row vectors

Run the thesis-statistics suite (`src/benchmarks.jl`) for ONE material, returning
the raw `NamedTuple` rows (one vector per table) rather than writing them:

  * `storage`       — data file: bytes on disk, in memory, sample count.
  * `lut`           — universal 4D-LUT size + generation time at a few resolutions.
  * `model_storage` — each best-fit model: parameter count and struct size.
  * `eval_timing`   — ns per `evaluate`/`lookup`, for each model + measured lookup.

Returning rows (instead of files) is what lets the multi-material merged tables
reuse these numbers — they come from BenchmarkTools' `@belapsed` and cost a few
seconds per material, so they must never be computed twice.
"""
function material_stats(data::MeasuredBRDF, material_path::String,
                        best_results::Vector{FitResult}; n_dirs::Int = 4096)
    storage = storage_stats([material_path])
    lut     = lut_storage_stats(data)
    models  = Any[r.model for r in best_results]
    model_storage = NamedTuple[]
    eval_timing   = NamedTuple[]
    if !isempty(models)
        model_storage = model_storage_stats(models = models)
        entries = Any[models...]; push!(entries, data)   # models + the measured lookup
        eval_timing = evaluate_benchmark(entries; n_dirs = n_dirs)
    end
    return (storage = storage, lut = lut,
            model_storage = model_storage, eval_timing = eval_timing)
end

"""
    write_stat_tables(statsdir, stats) -> Vector{String}

Write each table of a `material_stats` result to `<statsdir>/<table>.{csv,md,tex}`.
"""
function write_stat_tables(statsdir::String, stats::NamedTuple)
    mkpath(statsdir)
    written = String[]
    for key in keys(stats)
        append!(written, write_stats(getfield(stats, key), joinpath(statsdir, String(key))))
    end
    return written
end

"""
    write_merged_stat_tables(statsdir, per_material) -> Vector{String}

Concatenate the per-material `material_stats` results into one set of tables.
`per_material` is a vector of `(material_name, stats)` pairs. `storage` rows
already carry a `material` field, so they concatenate directly; the other three
are per-material but don't name it, so a `material` column is prepended.
"""
function write_merged_stat_tables(statsdir::String, per_material::Vector)
    mkpath(statsdir)
    written = String[]
    storage = NamedTuple[]
    for (_, s) in per_material
        append!(storage, s.storage)
    end
    append!(written, write_stats(storage, joinpath(statsdir, "storage")))

    for key in (:lut, :model_storage, :eval_timing)
        rows = NamedTuple[]
        for (name, s) in per_material
            for r in getfield(s, key)
                push!(rows, merge((material = name,), r))
            end
        end
        append!(written, write_stats(rows, joinpath(statsdir, String(key))))
    end
    return written
end

# ============================================================================
# Tables (CSV / Markdown / LaTeX)
# ============================================================================

_fmt(x) = isnan(x) ? "--" : @sprintf("%.4f", x)

"""
    write_tables(results, varied, material, cfg_line, fixed, out;
                 labels=nothing, stem="metrics") -> ranked

Write `<out>/<stem>.{csv,md,tex}` for `results`, ranked by RMS-LogWE.

`labels` turns this into the MERGED (multi-material) table: pass one material
name per entry of `results` and a leading "Material" column is added, with rows
grouped by material and ranked within each. Left `nothing` (the single-material
case) the output is exactly as if the parameter didn't exist.
"""
function write_tables(results::Vector{FitResult}, varied::Vector{Symbol},
                      material::String, cfg_line::String, fixed::Vector{Tuple{String,String}},
                      out::String; labels::Union{Nothing, Vector{String}} = nothing,
                      stem::String = "metrics")
    show_mat = labels !== nothing
    # Pair each result with its material label so sorting keeps the two aligned.
    paired = collect(zip(show_mat ? labels : fill("", length(results)), results))
    ranked_pairs = show_mat ?
        sort(paired; by = p -> (p[1], p[2].ok ? p[2].rmsle : Inf)) :
        sort(paired; by = p -> (p[2].ok ? p[2].rmsle : Inf))
    ranked = [p[2] for p in ranked_pairs]
    # Sentence-case for the Markdown line, lower-case lead-in for the LaTeX caption
    # (only the first letter differs — don't lowercase the whole thing, it holds "RMS-LogWE").
    sort_note = show_mat ? "Grouped by material, then sorted by RMS-LogWE" :
                           "Sorted by RMS-LogWE"
    caption_note = lowercase(sort_note[1:1]) * sort_note[2:end]

    # --- CSV: full record, every factor column ---
    open(joinpath(out, "$(stem).csv"), "w") do io
        println(io, (show_mat ? "material," : "") *
                    "model,optimizer,loss,sampling,samples,starts,cosine,arch,aniso,n_params," *
                    "rms_crwe,rmse,rmsle,ssim_mean,ssim_min,fit_seconds,status")
        for (mat, r) in ranked_pairs
            cells = String[]
            show_mat && push!(cells, mat)
            append!(cells, [
                r.name, r.run.optimizer, String(r.run.loss), String(r.run.sampling),
                string(r.run.samples), factor_str(r.run, :starts), factor_str(r.run, :cosine),
                factor_str(r.run, :arch), factor_str(r.run, :aniso), string(r.n_params),
                isnan(r.rms_crwe)  ? "" : @sprintf("%.6f", r.rms_crwe),
                isnan(r.rmse)      ? "" : @sprintf("%.6f", r.rmse),
                isnan(r.rmsle)     ? "" : @sprintf("%.6f", r.rmsle),
                isnan(r.ssim_mean) ? "" : @sprintf("%.6f", r.ssim_mean),
                isnan(r.ssim_min)  ? "" : @sprintf("%.6f", r.ssim_min),
                @sprintf("%.3f", isnan(r.seconds) ? 0.0 : r.seconds),
                r.ok ? "ok" : "failed"])
            println(io, join(cells, ","))
        end
    end

    # Column headers for MD/TeX: [Material] + Model + varied factors + metrics.
    varied_labels = [FACTOR_LABEL[k] for k in varied]
    lead = show_mat ? ["Material"] : String[]

    # --- Markdown ---
    open(joinpath(out, "$(stem).md"), "w") do io
        println(io, "# Model fits for `$material`\n")
        println(io, cfg_line, "\n")
        isempty(fixed) || println(io, "Fixed: ",
                                  join(["$l = $v" for (l, v) in fixed], ", "), "\n")
        println(io, "$(sort_note) (lower is better).\n")
        cols = vcat(lead, "Model", varied_labels,
                    ["Params", "RMS-LogWE", "RMS-CRWE", "RMSE", "SSIM (mean)", "SSIM (min)", "Fit [s]"])
        aligns = vcat(fill("---", length(lead) + 1 + length(varied_labels)), fill("--:", 7))
        println(io, "| ", join(cols, " | "), " |")
        println(io, "|", join(aligns, "|"), "|")
        for (mat, r) in ranked_pairs
            fvals = [factor_str(r.run, k) for k in varied]
            p = r.n_params < 0 ? "--" : string(r.n_params)
            status = r.ok ? "" : " ⚠️ failed"
            row = vcat(show_mat ? [mat] : String[], "$(r.name)$(status)", fvals,
                       [p, _fmt(r.rmsle), _fmt(r.rms_crwe), _fmt(r.rmse),
                        _fmt(r.ssim_mean), _fmt(r.ssim_min),
                        isnan(r.seconds) ? "--" : @sprintf("%.2f", r.seconds)])
            println(io, "| ", join(row, " | "), " |")
        end
    end

    # --- LaTeX (plain tabular + \hline, matching results.tex house style) ---
    open(joinpath(out, "$(stem).tex"), "w") do io
        safe = replace(material, "_" => raw"\_")
        label_slug = replace(material, r"[^a-zA-Z0-9]" => "-")
        ncol = length(lead) + 1 + length(varied) + 6   # [Material] + Model + varied + 6 metrics
        colspec = "l" * repeat("r", ncol - 1)
        headers = vcat(["\\textbf{$(l)}" for l in lead], "\\textbf{Model}",
                       ["\\textbf{$(l)}" for l in varied_labels],
                       ["\\textbf{Params}", "\\textbf{RMS-LogWE}", "\\textbf{RMS-CRWE}",
                        "\\textbf{RMSE}",
                        "\\textbf{SSIM}", "\\textbf{SSIM\\textsubscript{min}}"])
        println(io, "% Auto-generated by tools/fit_material.jl")
        println(io, "\\begin{table}[h]")
        println(io, "\\centering")
        println(io, "\\begin{tabular}{$colspec}")
        println(io, "\\hline")
        println(io, join(headers, " & "), " \\\\")
        println(io, "\\hline")
        for (mat, r) in ranked_pairs
            nm = replace(r.name, "_" => raw"\_")
            fvals = [replace(factor_str(r.run, k), "_" => raw"\_") for k in varied]
            p = r.n_params < 0 ? "--" : string(r.n_params)
            cells = vcat(show_mat ? [replace(mat, "_" => raw"\_")] : String[],
                         nm, fvals, [p, _fmt(r.rmsle), _fmt(r.rms_crwe), _fmt(r.rmse),
                                     _fmt(r.ssim_mean), _fmt(r.ssim_min)])
            println(io, join(cells, " & "), " \\\\")
        end
        println(io, "\\hline")
        println(io, "\\end{tabular}")
        fixed_txt = isempty(fixed) ? "" :
            " Fixed: " * join(["$l $v" for (l, v) in fixed], ", ") * "."
        println(io, "\\caption[Model fits for $safe]{Model fits for \\texttt{$safe}, ",
                    "$(caption_note).$fixed_txt}")
        println(io, "\\label{tab:fits-$(label_slug)}")
        println(io, "\\end{table}")
    end
    return ranked
end

# ============================================================================
# Merged (multi-material) tables
# ============================================================================

"""
    write_merged(records, parent, label, with_stats)

Merge every material's runs into ONE bundle at `parent`:
`metrics_all.{csv,md,tex}` — all runs of all materials in a single table with a
leading Material column — and, with `--stats`, `stats/` tables concatenated the
same way. `records` are the `run_material` return values; the per-material
bundles under `<parent>/<material>/` are left untouched.
"""
function write_merged(records::Vector, parent::String, label::String, with_stats::Bool)
    isempty(records) && return nothing
    mkpath(parent)

    all_results = FitResult[]
    labels = String[]
    for rec in records, r in rec.results
        push!(all_results, r)
        push!(labels, rec.name)
    end

    varied = varied_factors(all_results)
    fixed  = fixed_factors(all_results, varied)
    cfg_line = "$(length(records)) materials. Swept: " *
               (isempty(varied) ? "(nothing — single config)" :
                join([FACTOR_LABEL[k] for k in varied], ", "))
    write_tables(all_results, varied, label, cfg_line, fixed, parent;
                 labels = labels, stem = "metrics_all")
    println("Merged tables: metrics_all.csv, metrics_all.md, metrics_all.tex in $(parent)")

    if with_stats
        per = Any[(rec.name, rec.stats) for rec in records if rec.stats !== nothing]
        if !isempty(per)
            write_merged_stat_tables(joinpath(parent, "stats"), per)
            println("Merged stats: storage / model_storage / eval_timing / lut (csv/md/tex) in ",
                    joinpath(parent, "stats"))
        end
    end
    return nothing
end

# ============================================================================
# CLI
# ============================================================================

function parse_args(argv)
    opts = Dict{String, String}()
    flags = Set{String}()
    positional = String[]
    known_flags = Set(["neural", "errormaps", "stats", "neural-gpu"])
    i = 1
    while i <= length(argv)
        a = argv[i]
        if startswith(a, "--")
            key = a[3:end]
            if key in known_flags
                push!(flags, key); i += 1
            else
                i + 1 <= length(argv) || error("missing value for --$key")
                opts[key] = argv[i + 1]; i += 2
            end
        else
            push!(positional, a); i += 1
        end
    end
    return positional, opts, flags
end

# Split a comma list, trimming blanks.
_splitlist(s) = [strip(x) for x in split(s, ',') if !isempty(strip(x))]

# Fixed (non-swept) neural knobs: CLI option → (NeuralBRDFTrainer kwarg, parser).
const _NEURAL_KW = [
    ("neural-hidden-act",  :hidden_act,    s -> Symbol(strip(s))),
    ("neural-out-act",     :out_act,       s -> Symbol(strip(s))),
    ("neural-optimizer",   :optimizer,     s -> Symbol(strip(s))),
    ("neural-lr",          :learning_rate, s -> parse(Float64, s)),
    ("neural-weight-decay",:weight_decay,  s -> parse(Float64, s)),
    ("neural-batch",       :batch_size,    s -> parse(Int, s)),
    ("neural-seed",        :seed,          s -> parse(Int, s)),
]

"""
    neural_kwargs_from(opts, flags) -> Vector{Pair{Symbol,Any}}

Collect the fixed neural-trainer knobs the user actually passed (anything not
given keeps the trainer's own default). `--neural-gpu` maps to `use_gpu=true`.
"""
function neural_kwargs_from(opts, flags)
    kw = Pair{Symbol, Any}[]
    for (opt, kwarg, parse_fn) in _NEURAL_KW
        haskey(opts, opt) && push!(kw, kwarg => parse_fn(opts[opt]))
    end
    "neural-gpu" in flags && push!(kw, :use_gpu => true)
    return kw
end

"""
    run_material(material_path, out_dir_fn, models, optimizers, losses, samplings,
                 sampleslist, startslist, cosines, neural, hiddens, epochs, anisos,
                 neural_kw, resolution, metric_n, lights, errormaps, render_mode,
                 write_stats) -> (name, results, stats)

Fit + render + write tables for ONE material. This is the whole single-material
pipeline (previously the body of `main`), factored out so it can be run once
per file when `main` is given a folder of materials. `out_dir_fn(matslug)`
builds the output directory from the loaded BRDF's (sanitized) name, since that
name — not the source filename — is what the original single-material layout
keyed on.

Returns the material's name, its `FitResult`s, and its `material_stats` rows
(`nothing` without `--stats`) so `main` can merge them across materials.
"""
function run_material(material_path, out_dir_fn, models, optimizers, losses, samplings,
                      sampleslist, startslist, cosines, neural, hiddens, epochs, anisos,
                      neural_kw, resolution, metric_n, lights, errormaps, render_mode,
                      write_stats)
    println("Loading material: ", material_path)
    data = load_brdf(material_path)
    data.status == :ok || error("failed to load material ($(data.status)): $(data.error_msg)")
    println("  loaded: $(data.name)  [$(data.source_format)]  status=$(data.status)")

    runs = build_runs(models, optimizers, losses, samplings, sampleslist, startslist,
                      cosines, neural, hiddens, epochs, anisos, neural_kw)
    println("Runs to fit: $(length(runs))  ",
            "(models=$(length(models)), optimizers=$(length(optimizers)), ",
            "losses=$(length(losses)), samplings=$(length(samplings)), ",
            "samples=$(length(sampleslist)), starts=$(length(startslist)), ",
            "cosine=$(length(cosines))",
            neural ? ", neural-archs=$(length(hiddens)), neural-aniso=$(length(anisos))" : "", ")")

    matslug = replace(data.name, r"[^a-zA-Z0-9]" => "-")
    out = out_dir_fn(matslug)
    mkpath(out)
    println("Output: ", out)

    # Fit every run
    results = FitResult[]
    for (i, run) in enumerate(runs)
        tag = run_slug(run, FACTORS)   # verbose tag for progress (all factors)
        print("  [$(i)/$(length(runs))] $(tag) ... "); flush(stdout)
        r = fit_one(run, data, metric_n)
        push!(results, r)
        if r.ok
            @printf("RMS-LogWE=%.5f  SSIM=%.3f  (%.1fs)\n", r.rmsle, r.ssim_mean, r.seconds)
        else
            println("FAILED (", first(split(r.err, '\n')), ")")
        end
    end

    varied = varied_factors(results)
    fixed  = fixed_factors(results, varied)
    slugs  = assign_slugs(results, varied)

    # Renders
    if render_mode == "none"
        println("Rendering measured reference only (--render none) ...")
    else
        println("Rendering spheres ($(resolution)px, $(length(lights)) light(s), mode=$(render_mode)) ...")
    end
    n_png = render_runs(results, slugs, data, lights, joinpath(out, "renders");
                        resolution = resolution, errormaps = errormaps, mode = render_mode)

    # Tables
    cfg_line = "Swept: " * (isempty(varied) ? "(nothing — single config)" :
               join([FACTOR_LABEL[k] for k in varied], ", "))
    ranked = write_tables(results, varied, data.name, cfg_line, fixed, out)

    # Thesis statistics (opt-in): storage / eval-timing / LUT tables for this material
    stats_written = String[]
    stats = nothing
    if write_stats
        println("Computing thesis statistics (storage / eval timing / LUT) ...")
        stats = material_stats(data, material_path, best_ok_results(results); n_dirs = metric_n)
        stats_written = write_stat_tables(joinpath(out, "stats"), stats)
    end

    # Console summary
    println("\n=== Ranking (by RMS-LogWE) ===")
    varied_labels = [FACTOR_LABEL[k] for k in varied]
    hdr = vcat(rpad("model", 26), [lpad(l, 15) for l in varied_labels],
               [lpad("params", 8), lpad("RMS-LogWE", 10), lpad("SSIM", 8), lpad("SSIMmin", 8)])
    println(join(hdr, ""))
    for r in ranked
        fvals = [lpad(factor_str(r.run, k), 15) for k in varied]
        row = vcat(rpad(r.name, 26), fvals,
                   [lpad(r.n_params < 0 ? "--" : string(r.n_params), 8),
                    lpad(_fmt(r.rmsle), 10), lpad(_fmt(r.ssim_mean), 8),
                    lpad(_fmt(r.ssim_min), 8)])
        println(join(row, ""))
    end
    println("\nDone. $(n_png) JPEG(s) in $(joinpath(out, "renders"))")
    println("Tables: metrics.csv, metrics.md, metrics.tex in $(out)")
    if !isempty(stats_written)
        println("Stats: storage / model_storage / eval_timing / lut (csv/md/tex) in $(joinpath(out, "stats"))")
    end
    return (name = data.name, results = results, stats = stats)
end

function main(argv)
    positional, opts, flags = parse_args(argv)
    isempty(positional) && error("usage: julia --project=. tools/fit_material.jl <material|folder> [options]")

    # The given path is the whole truth: a folder ⇒ try every file in it, anything
    # else ⇒ try it as-is. Whether a file is a valid measured BRDF is decided by
    # `load_brdf` alone (a failure just skips that file).
    arg = positional[1]
    materials = isdir(arg) ? sort([p for p in readdir(arg; join = true) if isfile(p)]) : [arg]
    length(materials) > 1 && println("Folder input: $(length(materials)) item(s) to try")

    # Swept factor lists ("all" expands where supported) — shared across every material
    optimizers = resolve_optimizers(get(opts, "optimizer", "lm"))
    losses     = resolve_losses(get(opts, "loss", "cuberoot"))
    samplings  = resolve_samplings(get(opts, "sampling", "rusinkiewicz"))
    sampleslist = parse.(Int, _splitlist(get(opts, "samples", "2048")))
    startslist  = parse.(Int, _splitlist(get(opts, "starts", "1")))
    cosines     = resolve_bools(get(opts, "cosine-weight", "true"))
    models      = resolve_models(get(opts, "models", "all"))

    # Non-swept knobs
    resolution = parse(Int, get(opts, "resolution", "400"))
    metric_n   = parse(Int, get(opts, "metric-samples", "4096"))
    lights     = resolve_lights(get(opts, "lights", "ssim"))
    errormaps  = "errormaps" in flags
    render_mode = lowercase(get(opts, "render", "all"))
    render_mode in ("all", "best", "none") ||
        error("--render must be all|best|none, got \"$render_mode\"")
    write_stats = "stats" in flags

    neural = "neural" in flags
    hiddens = [parse.(Int, _splitlist(a)) for a in split(get(opts, "neural-hidden", "21,21"), ';')]
    epochs  = parse(Int, get(opts, "neural-epochs", "150"))
    anisos  = resolve_bools(get(opts, "neural-anisotropic", "false"))
    neural_kw = neural_kwargs_from(opts, flags)

    # Output layout. ONE material: <out>, or exports/fits/<matslug>_<timestamp>.
    # SEVERAL: a shared parent (<out>, or exports/fits/<folder>_<timestamp>) holding
    # one <matslug>/ subdir per material — they each need their own or they'd
    # overwrite each other's renders — plus the merged tables at the parent.
    stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    out_root = get(opts, "out", "")
    default_root = normpath(joinpath(@__DIR__, "..", "exports", "fits"))
    many = length(materials) > 1
    dirslug = replace(basename(rstrip(normpath(arg), ['/', '\\'])), r"[^a-zA-Z0-9]" => "-")
    parent = normpath(isempty(out_root) ? joinpath(default_root, "$(dirslug)_$(stamp)") : out_root)
    last_out = Ref("")
    out_dir_fn = matslug -> (last_out[] = many ? joinpath(parent, matslug) :
        normpath(isempty(out_root) ? joinpath(default_root, "$(matslug)_$(stamp)") : out_root))

    failed = String[]
    records = Any[]
    for material_path in materials
        try
            rec = run_material(material_path, out_dir_fn, models, optimizers, losses, samplings,
                        sampleslist, startslist, cosines, neural, hiddens, epochs, anisos,
                        neural_kw, resolution, metric_n, lights, errormaps, render_mode,
                        write_stats)
            push!(records, rec)
        catch e
            @warn "material failed, skipping" material = material_path exception = e
            push!(failed, material_path)
        end
    end

    if many
        ok = length(materials) - length(failed)
        println("\nDone: $(ok)/$(length(materials)) material(s) fit successfully")
        isempty(failed) || println("Skipped: ", join(basename.(failed), ", "))
        write_merged(records, parent, dirslug, write_stats)
        last_out[] = parent
    end
    return last_out[]
end

main(ARGS)
