"""
Thesis statistics: storage, timing, and accuracy benchmarks.

Turns the platform's registries and error metrics into thesis-ready tables.
Four benchmark groups, each returning a `Vector{NamedTuple}` (one row per
entry) that `write_stats` serialises to CSV, Markdown, and LaTeX:

1. `storage_stats(paths)`        — per data file: bytes on disk, bytes in
   memory, sample count. Feeds the MERL-vs-BiRD format comparison (RQ4).
2. `model_storage_stats()`       — per model: parameter count and struct size.
3. `evaluate_benchmark(entries)` — ns per `lookup`/`evaluate` call, measured
   over the deterministic metric lattice.
4. `fit_benchmark(datas, model_types, optimizers)` — fit time plus the accuracy
   metrics from `analysis.jl` (RMS-LogWE, RMS-CRWE, RMSE, image SSIM) for every
   material x model x optimizer combination.

`run_thesis_benchmarks(paths)` orchestrates all of them and writes the tables
to `.thesis/stats/` by default.

Timing methodology: all timings come from BenchmarkTools.jl (`@belapsed`), the
standard Julia micro-benchmarking package. It auto-tunes the evaluation count,
runs a warmup so JIT compilation is excluded, takes many samples, and reports
the *minimum* — the noise-robust estimator for deterministic kernels. Cheap
per-call kernels (`evaluate`/`lookup`, LUT generation) use the default budget; the
expensive fits use a capped `samples`/`seconds` budget so a large sweep stays
tractable.
"""

using StaticArrays
using BenchmarkTools

# ============================================================================
# Formatting / table writers
# ============================================================================

_fmt_cell(x::AbstractFloat) = isfinite(x) ? string(round(x, sigdigits = 4)) : string(x)
_fmt_cell(x) = string(x)

_latex_escape(s::String) = replace(s, "_" => "\\_", "%" => "\\%", "&" => "\\&")

"""
    write_stats(rows, basepath; formats=(:csv, :md, :tex)) -> Vector{String}

Serialise a vector of NamedTuple rows (all with identical keys) to
`basepath.csv` / `.md` / `.tex`. Returns the written file paths.
The `.tex` output is a booktabs tabular ready for `\\input{}` in the thesis.
"""
function write_stats(rows::Vector{<:NamedTuple}, basepath::String;
                     formats = (:csv, :md, :tex))
    isempty(rows) && return String[]
    mkpath(dirname(abspath(basepath)))
    header = String.(collect(keys(rows[1])))
    body = [[_fmt_cell(v) for v in values(r)] for r in rows]
    written = String[]

    if :csv in formats
        path = basepath * ".csv"
        open(path, "w") do io
            println(io, join(header, ","))
            for row in body
                println(io, join(row, ","))
            end
        end
        push!(written, path)
    end

    if :md in formats
        path = basepath * ".md"
        open(path, "w") do io
            println(io, "| ", join(header, " | "), " |")
            println(io, "|", repeat("---|", length(header)))
            for row in body
                println(io, "| ", join(row, " | "), " |")
            end
        end
        push!(written, path)
    end

    if :tex in formats
        path = basepath * ".tex"
        open(path, "w") do io
            println(io, "\\begin{tabular}{", repeat("l", length(header)), "}")
            println(io, "\\toprule")
            println(io, join(_latex_escape.(header), " & "), " \\\\")
            println(io, "\\midrule")
            for row in body
                println(io, join(_latex_escape.(row), " & "), " \\\\")
            end
            println(io, "\\bottomrule")
            println(io, "\\end{tabular}")
        end
        push!(written, path)
    end

    return written
end

# ============================================================================
# 1. Storage: measured data files
# ============================================================================

"""
    storage_stats(paths) -> Vector{NamedTuple}

Per data file: source format, bytes on disk, bytes of the in-memory
representation (`Base.summarysize` of the loaded `MeasuredBRDF`), and the
sample count. Files that fail to load are reported with their error message.
"""
function storage_stats(paths::Vector{String})
    rows = NamedTuple[]
    for path in paths
        brdf = load_brdf(path)
        if brdf.status == :failed
            @warn "Skipping (failed to load)" path error = brdf.error_msg
            continue
        end
        push!(rows, (
            material   = brdf.name,
            format     = brdf.source_format,
            file_MB    = filesize(path) / 1024^2,
            memory_MB  = Base.summarysize(brdf) / 1024^2,
            n_samples  = n_samples_str(brdf._data),
        ))
    end
    return rows
end

# ============================================================================
# 2. Storage: fitted representations (analytical params, neural weights, LUT)
# ============================================================================

# Instantiate a model from its fit spec's initial guess; `nothing` if the
# model has no spec (e.g. NeuralBRDFModel, which is produced by its trainer).
function _default_instance(M::Type{<:BRDFModel})
    try
        spec = _fit_spec(M)
        return spec.reconstruct(spec.p0)
    catch
        return nothing
    end
end

# Base (un-parameterised) type of a model instance: `_fit_spec` is defined on the
# UnionAll (e.g. `LambertianModel`), not the concrete `LambertianModel{Float64}`.
_base_type(model) = Base.typename(typeof(model)).wrapper

# Neural models carry weight matrices (Ws/bs) instead of a fit spec; detect by
# field rather than by type — NeuralBRDFModel is only defined once plugins load.
function _n_parameters(model::BRDFModel)
    if hasproperty(model, :Ws)
        return sum(length, model.Ws) + sum(length, model.bs)
    end
    return length(_fit_spec(_base_type(model)).p0)
end

"""
    model_storage_stats(; models=nothing) -> Vector{NamedTuple}

Parameter count and in-memory size for each model. By default covers every
registered model that has a fit spec; pass `models` (a Vector of instances,
e.g. fitted or trained ones) to measure those instead — required for
`NeuralBRDFModel`, which only exists once trained.
"""
function model_storage_stats(; models::Union{Nothing, Vector} = nothing)
    if models === nothing
        models = Any[]
        for name in sort!(collect(keys(MODEL_REGISTRY)))
            inst = _default_instance(MODEL_REGISTRY[name])
            inst === nothing || push!(models, inst)
        end
    end
    return NamedTuple[(
        model    = string(nameof(typeof(m))),
        n_params = _n_parameters(m),
        bytes    = Base.summarysize(m),
    ) for m in models]
end

"""
    lut_storage_stats(brdf; dims_list, dtypes) -> Vector{NamedTuple}

Size and generation time of the universal 4D LUT for several resolutions and
precisions — the storage cost of the *rendering* representation, as opposed
to the raw data (`storage_stats`) or the fitted parameters
(`model_storage_stats`).
"""
function lut_storage_stats(brdf::BRDFData;
                           dims_list = [(32, 32, 32, 1), (64, 64, 64, 1)],
                           dtypes = (:float16, :float32))
    rows = NamedTuple[]
    for dims in dims_list, dtype in dtypes
        # Generating one LUT takes ~0.1-1s; cap the benchmark budget so the sweep is quick.
        t = @belapsed generate_lut($brdf; dims = $dims, dtype = $dtype) samples = 3 seconds = 20 evals = 1
        n_texels = prod(dims)
        bytes_per = dtype == :float16 ? 2 : 4
        push!(rows, (
            dims        = join(dims, "x"),
            dtype       = dtype,
            size_MB     = n_texels * 3 * bytes_per / 1024^2,
            gen_time_s  = t,
        ))
    end
    return rows
end

# ============================================================================
# 3. Compute time of evaluate / lookup
# ============================================================================

# Sum `lookup` over a fixed direction lattice. Returning the accumulator keeps
# the compiler from eliding the loop; this is the kernel handed to `@belapsed`.
function _lookup_loop(f, wis, wos)
    acc = 0.0f0
    @inbounds for k in eachindex(wis)
        acc += lookup(f, wis[k], wos[k]).r
    end
    return acc
end

"""
    evaluate_benchmark(entries; n_dirs=4096) -> Vector{NamedTuple}

Nanoseconds per BRDF evaluation for each entry (a `BRDFModel` or a
`MeasuredBRDF` — both go through the unified `lookup`), over `n_dirs`
deterministic direction pairs. Timed with BenchmarkTools' `@belapsed`
(minimum over auto-tuned samples) on the whole lattice, divided by `n_dirs`.

If `entries` is `nothing`, benchmarks the default instance of every
registered model with a fit spec.
"""
function evaluate_benchmark(entries = nothing; n_dirs::Int = 4096)
    if entries === nothing
        entries = Any[]
        for name in sort!(collect(keys(MODEL_REGISTRY)))
            inst = _default_instance(MODEL_REGISTRY[name])
            inst === nothing || push!(entries, inst)
        end
    end
    wis, wos = _metric_directions(n_dirs)
    rows = NamedTuple[]
    for e in entries
        # Interpolate ($) the kernel args so globals don't taint the timing.
        secs = @belapsed _lookup_loop($e, $wis, $wos)
        ns = secs * 1e9 / n_dirs
        label = e isa MeasuredBRDF ? "$(e.name) ($(e.source_format))" :
                                     string(nameof(typeof(e)))
        push!(rows, (
            entry        = label,
            ns_per_eval  = ns,
            Meval_per_s  = 1000.0 / ns,
        ))
    end
    return rows
end

# ============================================================================
# 4. Fit time + accuracy
# ============================================================================

"""
    fit_benchmark(datas, model_types, optimizers;
                  ssim=true, ssim_resolution=256, metric_n=4096,
                  time_fit=true, fit_samples=5, fit_seconds=15) -> Vector{NamedTuple}

Fit every model type with every optimizer to every material, recording fit
time and the accuracy metrics from `analysis.jl`:

- `rmse`      — linear luminance RMSE (BRDF space)
- `rms_crwe`  — cube-root-weighted RMSE
- `rmsle`     — RMS-LogWE, log-compressed, the headline perceptual metric
  (weights the diffuse lobe over the peak)
- `ssim_mean` / `ssim_min` — image-space SSIM over the standard light set
  (skipped when `ssim=false`; it dominates runtime)

Fit time is the minimum from BenchmarkTools' `@belapsed`, capped at
`fit_samples` samples / `fit_seconds` seconds per combination (repeated fits
are costly, so the budget is bounded rather than default). One real fit is
always run outside the benchmark to produce the model for the accuracy
metrics, which also warms up JIT. Set `time_fit=false` to skip the repeated
timing entirely and report just that single fit's wall-clock time.

Combinations that error (e.g. an optimizer that does not support a model) are
recorded with `status = :failed` instead of aborting the sweep.
"""
function fit_benchmark(datas::Vector{MeasuredBRDF},
                       model_types::Vector,
                       optimizers::Vector;
                       ssim::Bool = true, ssim_resolution::Int = 256,
                       metric_n::Int = 4096, time_fit::Bool = true,
                       fit_samples::Int = 5, fit_seconds::Real = 15)
    rows = NamedTuple[]
    for data in datas
        data.status == :failed && continue
        for opt in optimizers, M in model_types
            t = NaN
            model = nothing
            status = :ok
            try
                # One real fit: yields the model for accuracy metrics + warms JIT.
                dt = @elapsed model = fit_brdf(opt, M, data)
                t = time_fit ?
                    (@belapsed fit_brdf($opt, $M, $data) samples = fit_samples seconds = fit_seconds evals = 1) :
                    dt
            catch e
                @warn "Fit failed" material = data.name model = M optimizer = typeof(opt) exception = e
                status = :failed
            end
            if model === nothing
                push!(rows, (material = data.name, model = string(nameof(M)),
                             optimizer = string(nameof(typeof(opt))),
                             fit_time_s = t, rmse = NaN, rms_crwe = NaN, rmsle = NaN,
                             ssim_mean = NaN, ssim_min = NaN, status = status))
                continue
            end
            rmse = brdf_rmse(data, model; n = metric_n)
            crwe = brdf_rms_crwe(data, model; n = metric_n)
            rmsle = brdf_rmsle(data, model; n = metric_n)
            sm, smin = NaN, NaN
            if ssim
                s = image_ssim(data, model; resolution = ssim_resolution)
                sm, smin = s.mean, s.min
            end
            push!(rows, (material = data.name, model = string(nameof(M)),
                         optimizer = string(nameof(typeof(opt))),
                         fit_time_s = t, rmse = rmse, rms_crwe = crwe, rmsle = rmsle,
                         ssim_mean = sm, ssim_min = smin, status = status))
        end
    end
    return rows
end

# ============================================================================
# Orchestrator
# ============================================================================

# Registered model types that have a fit spec (fittable analytical models).
function _fittable_model_types()
    # Registry values are the (UnionAll) model types, e.g. `AshikhminShirleyModel`
    # for the parametric `AshikhminShirleyModel{T}` — hence `Type[]`, not `DataType[]`.
    types = Type[]
    for name in sort!(collect(keys(MODEL_REGISTRY)))
        M = MODEL_REGISTRY[name]
        _default_instance(M) === nothing || push!(types, M)
    end
    return types
end

"""
    run_thesis_benchmarks(paths;
                          outdir=".thesis/stats",
                          model_types=_fittable_model_types(),
                          optimizers=[LevenbergMarquardt()],
                          lut=true, kwargs...) -> Dict{Symbol, Vector}

Run the full statistics suite over the given data files and write
`storage`, `model_storage`, `eval_timing`, `lut` and `fits` tables (CSV +
Markdown + LaTeX each) into `outdir`. Extra `kwargs` are forwarded to
`fit_benchmark` (e.g. `ssim=false` for a quick pass).

Returns the raw row vectors keyed by table name for interactive use.
"""
function run_thesis_benchmarks(paths::Vector{String};
                               outdir::String = ".thesis/stats",
                               model_types::Vector = _fittable_model_types(),
                               optimizers::Vector = [LevenbergMarquardt()],
                               lut::Bool = true,
                               kwargs...)
    mkpath(outdir)
    results = Dict{Symbol, Vector}()

    @info "[1/5] Storage: data files"
    results[:storage] = storage_stats(paths)
    write_stats(results[:storage], joinpath(outdir, "storage"))

    @info "[2/5] Storage: model parameters"
    results[:model_storage] = model_storage_stats()
    write_stats(results[:model_storage], joinpath(outdir, "model_storage"))

    datas = MeasuredBRDF[b for b in load_brdf.(paths) if b.status != :failed]

    @info "[3/5] Evaluate timing (models + measured lookup)"
    entries = Any[]
    for name in sort!(collect(keys(MODEL_REGISTRY)))
        inst = _default_instance(MODEL_REGISTRY[name])
        inst === nothing || push!(entries, inst)
    end
    append!(entries, datas)
    results[:eval_timing] = evaluate_benchmark(entries)
    write_stats(results[:eval_timing], joinpath(outdir, "eval_timing"))

    if lut && !isempty(datas)
        @info "[4/5] Universal LUT storage/generation time (first material)"
        results[:lut] = lut_storage_stats(datas[1])
        write_stats(results[:lut], joinpath(outdir, "lut"))
    else
        @info "[4/5] LUT benchmark skipped"
    end

    @info "[5/5] Fit time + accuracy ($(length(datas)) materials x $(length(model_types)) models x $(length(optimizers)) optimizers)"
    results[:fits] = fit_benchmark(datas, model_types, optimizers; kwargs...)
    write_stats(results[:fits], joinpath(outdir, "fits"))

    @info "Done. Tables written to $outdir"
    return results
end

export storage_stats, model_storage_stats, lut_storage_stats
export evaluate_benchmark, fit_benchmark
export run_thesis_benchmarks, write_stats
