#!/usr/bin/env julia
"""
RQ5 review bundle builder.

For every material in the selected datasets this fits **every registered
analytical model** plus the neural MLP (both feature mappings), keeps the best
of each family by mean SSIM, and writes a self-contained review bundle:

    <out>/records.jsonl                       one JSON record per material
    <out>/renders/<dataset>__<material>/
        measured__g1.jpg      measured__g5.jpg
        analytical__g1.jpg    analytical__g5.jpg
        neural__g1.jpg        neural__g5.jpg

Renders use a single light direction (θ=45°, φ=60°, the middle SSIM light) at
two exposures: the normal gain π and a 5× boost so dark materials stay
judgeable. SSIM itself is still averaged over all three SSIM lights.

`records.jsonl` is appended one line at a time, so an interrupted run loses at
most the material in flight; re-running skips whatever is already recorded.

Usage
-----
    julia --project=. tools/rq5_gather.jl [options]

    --datasets  all          all | comma list of merl,utia,rgl_iso,rgl_aniso
    --out       exports/rq5_review
    --limit     0            stop after N materials per dataset (0 = no limit)
    --resolution 400         render resolution before cropping
    --fresh                  ignore existing records.jsonl and start over
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

using BRDFPlatform
using BRDFPlatform: MODEL_REGISTRY, OPTIMIZER_REGISTRY, MeasuredBRDF, BRDFModel,
                    NeuralBRDFModel, NeuralBRDFTrainer, load_brdf, fit_brdf,
                    evaluate, display_parameters, render_brdf_sphere, image_ssim,
                    brdf_rms_crwe, brdf_rmse, brdf_rmsle
using JSON3
using JpegTurbo: jpeg_encode
using Colors
using Printf
using Dates

# ============================================================================
# Configuration
# ============================================================================

# Fixed fit setup. `uniform_cosine` sampling is the best-performing variant for
# 269/312 materials in the existing sweeps, so it is held fixed here to keep the
# 13-model sweep affordable.
const OPT_NAME   = "lbfgsb"
const LOSS       = :log
const SAMPLING   = :uniform_cosine
const N_SAMPLES  = 8192
const N_STARTS   = 1
const COSINE     = true
const METRIC_N   = 4096

const NEURAL_HIDDEN = [21, 21]
const NEURAL_EPOCHS = 150

# Single comparison light: SSIM_LIGHT_DIRS[2] == (π/4, π/3) == "th045_ph060".
const LIGHT_THETA = Float32(π / 4)
const LIGHT_PHI   = Float32(π / 3)

# Exposure variants: (suffix, gain).
const GAINS = [("g1", Float64(π)), ("g5", 5 * Float64(π))]

const CROP_MARGIN  = 70
const JPEG_QUALITY = 80

const DATASET_DIRS = [
    "merl"       => joinpath("test", "data", "merl"),
    "utia"       => joinpath("test", "data", "utia"),
    "rgl"        => joinpath("test", "data", "RGL"),
]

# Data files only — the dataset folders also hold readmes and desktop.ini.
const DATA_EXTS = Set([".binary", ".bsdf", ".bin"])

# ============================================================================
# Small helpers
# ============================================================================

function crop_margin(img::AbstractMatrix, margin::Int)
    h, w = size(img)
    (margin <= 0 || h <= 2margin || w <= 2margin) && return img
    return img[margin+1:h-margin, margin+1:w-margin]
end

save_render(path, img) = (jpeg_encode(path, crop_margin(img, CROP_MARGIN);
                                      quality = JPEG_QUALITY); path)

_is_neural(name::AbstractString) = occursin("neural", lowercase(name))

"Every registered analytical model type, sorted by name for a stable order."
function analytical_model_types()
    names = sort([String(k) for k in keys(MODEL_REGISTRY) if !_is_neural(String(k))])
    return [MODEL_REGISTRY[Symbol(n)] for n in names]
end

function n_params_of(model::BRDFModel, T::Type)
    model isa NeuralBRDFModel &&
        return sum(length, model.Ws; init = 0) + sum(length, model.bs; init = 0)
    try
        return length(BRDFPlatform._fit_spec(T).p0)
    catch
        return -1
    end
end

"`display_parameters` coerced into something JSON3 can always write."
function json_params(model::BRDFModel)
    out = Dict{String, Any}()
    for (k, v) in display_parameters(model)
        out[k] = v isa AbstractVector ? collect(Float64.(v)) :
                 v isa Real           ? Float64(v) :
                 v isa Bool           ? v : string(v)
    end
    return out
end

# ============================================================================
# Fitting
# ============================================================================

struct Candidate
    family::String              # "analytical" | "neural"
    name::String
    variant::String             # extra config that distinguishes sibling runs
    model::Union{BRDFModel, Nothing}
    n_params::Int
    rms_logwe::Float64
    rms_crwe::Float64
    rmse::Float64
    ssim_mean::Float64
    ssim_min::Float64
    seconds::Float64
    ok::Bool
    err::String
end

function evaluate_fit(family, name, variant, T, opt, data)
    try
        model = nothing
        secs  = @elapsed (model = fit_brdf(opt, T, data))
        ss    = image_ssim(data, model)
        return Candidate(family, name, variant, model, n_params_of(model, T),
                         brdf_rmsle(data, model; n = METRIC_N),
                         brdf_rms_crwe(data, model; n = METRIC_N),
                         brdf_rmse(data, model; n = METRIC_N),
                         ss.mean, ss.min, secs, true, "")
    catch e
        return Candidate(family, name, variant, nothing, -1,
                         NaN, NaN, NaN, NaN, NaN, NaN, false, sprint(showerror, e))
    end
end

function fit_all_analytical(data::MeasuredBRDF)
    out = Candidate[]
    for T in analytical_model_types()
        opt = OPTIMIZER_REGISTRY[Symbol(OPT_NAME)](; loss = LOSS, sampling = SAMPLING,
                                                     n_samples = N_SAMPLES,
                                                     n_starts = N_STARTS,
                                                     cosine_weight = COSINE)
        push!(out, evaluate_fit("analytical", String(nameof(T)), "--", T, opt, data))
    end
    return out
end

function fit_all_neural(data::MeasuredBRDF)
    out = Candidate[]
    for aniso in (false, true)
        trainer = NeuralBRDFTrainer(; hidden = NEURAL_HIDDEN, epochs = NEURAL_EPOCHS,
                                      anisotropic = aniso, loss = LOSS,
                                      sampling = SAMPLING, n_samples = N_SAMPLES,
                                      cosine_weight = COSINE)
        variant = aniso ? "aniso" : "iso"
        push!(out, evaluate_fit("neural", "NeuralBRDFModel", variant,
                                NeuralBRDFModel, trainer, data))
    end
    return out
end

"Highest mean SSIM among the runs that converged, or `nothing` if all failed."
function best_of(cands::Vector{Candidate})
    ok = filter(c -> c.ok && !isnan(c.ssim_mean), cands)
    isempty(ok) && return nothing
    return ok[argmax([c.ssim_mean for c in ok])]
end

# The three metrics a "best fit" can be selected by. SSIM is maximised, the two
# error metrics are minimised.
const SELECTORS = ["ssim" => (c -> c.ssim_mean, true),
                   "logwe" => (c -> c.rms_logwe, false),
                   "crwe" => (c -> c.rms_crwe, false)]

"Best candidate under one selector, skipping runs whose metric is NaN."
function best_by(cands::Vector{Candidate}, getter, maximise::Bool)
    ok = filter(c -> c.ok && !isnan(getter(c)), cands)
    isempty(ok) && return nothing
    vals = [getter(c) for c in ok]
    return ok[maximise ? argmax(vals) : argmin(vals)]
end

# Filename-safe identifier for one candidate, so a render can be traced back to
# the exact run that produced it.
function cand_slug(c::Candidate)
    base = "$(c.family)_$(c.name)"
    c.variant == "--" || (base *= "_$(c.variant)")
    return replace(base, r"[^A-Za-z0-9_]" => "")
end

# ============================================================================
# Rendering
# ============================================================================

function render_pair(target, dir::String, stem::String, resolution::Int)
    paths = Dict{String, String}()
    for (tag, gain) in GAINS
        img, _ = render_brdf_sphere(target, LIGHT_THETA, LIGHT_PHI;
                                    resolution = resolution, intensity = gain)
        file = "$(stem)__$(tag).jpg"
        save_render(joinpath(dir, file), img)
        paths[tag] = file
    end
    return paths
end

# ============================================================================
# Records
# ============================================================================

candidate_json(c::Candidate, renders) = Dict{String, Any}(
    "model"      => c.name,
    "variant"    => c.variant,
    "n_params"   => c.n_params,
    "params"     => c.model === nothing ? Dict{String, Any}() : json_params(c.model),
    "rms_logwe"  => c.rms_logwe,
    "rms_crwe"   => c.rms_crwe,
    "rmse"       => c.rmse,
    "ssim_mean"  => c.ssim_mean,
    "ssim_min"   => c.ssim_min,
    "fit_seconds"=> c.seconds,
    "renders"    => renders,
)

# Every converged candidate with its full metric set and fitted parameters.
# Storing all of it here is what lets a later analysis pick a different "best"
# metric without refitting anything.
ranking_json(cands::Vector{Candidate}) = [
    Dict{String, Any}("model" => c.name, "variant" => c.variant,
                      "family" => c.family, "n_params" => c.n_params,
                      "slug" => cand_slug(c),
                      "ssim_mean" => c.ssim_mean, "ssim_min" => c.ssim_min,
                      "rms_logwe" => c.rms_logwe, "rms_crwe" => c.rms_crwe,
                      "rmse" => c.rmse, "fit_seconds" => c.seconds,
                      "params" => c.model === nothing ? Dict{String, Any}() : json_params(c.model),
                      "ok" => c.ok)
    for c in sort(filter(c -> c.ok, cands), by = c -> -c.ssim_mean)
]

function already_done(path::String)
    done = Set{String}()
    isfile(path) || return done
    for line in eachline(path)
        isempty(strip(line)) && continue
        try
            push!(done, String(JSON3.read(line)["key"]))
        catch
        end
    end
    return done
end

# ============================================================================
# Per-material driver
# ============================================================================

function process_material(path::String, dataset::String, out::String, resolution::Int)
    data = load_brdf(path)
    data.status == :failed && error("load failed: $(something(data.error_msg, "unknown"))")

    material = splitext(basename(path))[1]
    key      = "$(dataset)__$(material)"

    analytical = fit_all_analytical(data)
    neural     = fit_all_neural(data)
    best_a     = best_of(analytical)
    best_n     = best_of(neural)
    (best_a === nothing && best_n === nothing) && error("every fit failed")

    rdir = joinpath(out, "renders", key)
    mkpath(rdir)
    measured_r = render_pair(data, rdir, "measured", resolution)
    a_r = best_a === nothing ? Dict{String,String}() :
          render_pair(best_a.model, rdir, "analytical", resolution)
    n_r = best_n === nothing ? Dict{String,String}() :
          render_pair(best_n.model, rdir, "neural", resolution)

    # Which run each metric would pick, per family, plus one render per distinct
    # winner. The two SSIM winners are already rendered under their legacy
    # names, so only genuinely new winners cost extra work.
    rendered = Dict{String, Any}()
    best_a === nothing || (rendered[cand_slug(best_a)] = a_r)
    best_n === nothing || (rendered[cand_slug(best_n)] = n_r)

    winners = Dict{String, Any}()
    for (family, cands) in ("analytical" => analytical, "neural" => neural)
        picks = Dict{String, Any}()
        for (mname, (getter, maximise)) in SELECTORS
            c = best_by(cands, getter, maximise)
            c === nothing && continue
            slug = cand_slug(c)
            picks[mname] = slug
            haskey(rendered, slug) ||
                (rendered[slug] = render_pair(c.model, rdir, slug, resolution))
        end
        winners[family] = picks
    end

    return Dict{String, Any}(
        "key"         => key,
        "material"    => material,
        "dataset"     => dataset,
        "source"      => path,
        "n_samples"   => BRDFPlatform.n_samples_str(data._data),
        "measured"    => Dict("renders" => measured_r),
        "analytical"  => best_a === nothing ? nothing : candidate_json(best_a, a_r),
        "neural"      => best_n === nothing ? nothing : candidate_json(best_n, n_r),
        "winners"     => winners,
        "candidates"  => rendered,
        "ranking"     => ranking_json(vcat(analytical, neural)),
        "generated"   => string(now()),
    )
end

# ============================================================================
# CLI
# ============================================================================

function parse_args(argv)
    opts = Dict("datasets" => "all", "out" => joinpath("exports", "rq5_review"),
                "limit" => "0", "resolution" => "400", "fresh" => "false")
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--fresh"
            opts["fresh"] = "true"; i += 1
        elseif startswith(a, "--") && i < length(argv)
            opts[a[3:end]] = argv[i+1]; i += 2
        else
            error("unrecognised argument: $a")
        end
    end
    return opts
end

function selected_datasets(spec::AbstractString)
    s = lowercase(strip(spec))
    s == "all" && return DATASET_DIRS
    want = Set(strip.(split(s, ',')))
    # rgl_iso / rgl_aniso both live in the single RGL folder
    return [p for p in DATASET_DIRS if p.first in want ||
            (p.first == "rgl" && any(startswith(w, "rgl") for w in want))]
end

function main(argv)
    opts       = parse_args(argv)
    out        = opts["out"]
    limit      = parse(Int, opts["limit"])
    resolution = parse(Int, opts["resolution"])
    mkpath(out)

    records = joinpath(out, "records.jsonl")
    opts["fresh"] == "true" && isfile(records) && rm(records)
    done = already_done(records)
    isempty(done) || @info "resuming, $(length(done)) material(s) already recorded"

    models = analytical_model_types()
    @info "analytical models" n = length(models) names = String.(nameof.(models))

    t0 = time()
    n_ok = n_skip = n_fail = 0
    for (dsname, dir) in selected_datasets(opts["datasets"])
        isdir(dir) || (@warn "missing dataset dir, skipping" dir; continue)
        files = sort(filter(readdir(dir; join = true)) do f
            isfile(f) && lowercase(splitext(f)[2]) in DATA_EXTS
        end)
        limit > 0 && (files = files[1:min(limit, end)])
        @info "dataset" dsname dir n_files = length(files)

        for (i, path) in enumerate(files)
            material = splitext(basename(path))[1]
            key = "$(dsname)__$(material)"
            if key in done
                n_skip += 1
                continue
            end
            @printf("[%s %3d/%3d] %-38s ", dsname, i, length(files), material)
            flush(stdout)
            try
                rec = process_material(path, dsname, out, resolution)
                open(records, "a") do io
                    JSON3.write(io, rec); println(io)
                end
                a = rec["analytical"]; n = rec["neural"]
                @printf("A:%-26s ssim %.4f | N:%-5s ssim %.4f\n",
                        a === nothing ? "-" : "$(a["model"])",
                        a === nothing ? NaN : a["ssim_mean"],
                        n === nothing ? "-" : n["variant"],
                        n === nothing ? NaN : n["ssim_mean"])
                n_ok += 1
            catch e
                println("FAILED: ", sprint(showerror, e))
                n_fail += 1
            end
        end
    end

    @printf("\ndone in %.1f min — %d fitted, %d skipped, %d failed\n",
            (time() - t0) / 60, n_ok, n_skip, n_fail)
    println("records: ", records)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
