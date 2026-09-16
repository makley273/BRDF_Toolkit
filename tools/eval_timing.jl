#!/usr/bin/env julia
"""
tools/eval_timing.jl — evaluation cost of every analytical model and of the
measured lookup in each supported format.

    julia --project=. tools/eval_timing.jl [--n-dirs 4096] [--out exports/eval_timing]

Two questions, one table each:

  * How expensive is one `evaluate` call per analytical model? This depends only
    on the model type, so one default instance each is enough.
  * How expensive is one measured `lookup` per format? This depends on the
    format's storage layout and interpolation scheme, so it is measured on
    several materials per format (five for MERL and UTIA, the full 62-material
    corpus for RGL, whose file size varies far more across materials — see
    `SETS` below). Within a format the spread across materials also shows
    whether file size affects lookup cost.

Timing comes from `evaluate_benchmark` (BenchmarkTools `@belapsed`, minimum over
auto-tuned samples, over a deterministic direction lattice), so it is the same
methodology as the thesis statistics export.
"""

using BRDFPlatform
using Printf
using Statistics

_argval(flag, default) = (i = findfirst(==(flag), ARGS);
                          i === nothing || i == length(ARGS) ? default : ARGS[i+1])

n_dirs = parse(Int, _argval("--n-dirs", "4096"))
outdir = _argval("--out", "exports/eval_timing")

# MERL and UTIA use a fixed sample grid per format, so file size barely varies
# across materials and five files span the format. RGL's tensor file grows
# with how peaked the material is (0.6 to 8.5 MB across the database), so five
# files would not necessarily represent the corpus the way they do for the
# other two formats: RGL is timed over every file instead.
const SETS = [
    ("MERL", [
        "test/data/merl/blue-metallic-paint2.binary",
        "test/data/merl/blue-rubber.binary",
        "test/data/merl/chrome.binary",
        "test/data/merl/gold-paint.binary",
        "test/data/merl/white-fabric.binary",
    ]),
    ("RGL", sort([joinpath("test/data/RGL", f) for f in readdir("test/data/RGL")
                  if endswith(f, ".bsdf")])),
    ("UTIA", [
        "test/data/utia/m001_3M_Scotchlite.bin",
        "test/data/utia/m071_fabric106.bin",
        "test/data/utia/m072_fabric107.bin",
        "test/data/utia/m106_leather01.bin",
        "test/data/utia/m126_plastic01.bin",
    ]),
]

mkpath(outdir)

# ---------------------------------------------------------------------------
# 1. Analytical models
# ---------------------------------------------------------------------------

_is_neural(name) = occursin("neural", lowercase(String(name)))

models = Any[]
for name in sort!(collect(keys(MODEL_REGISTRY)))
    _is_neural(name) && continue
    inst = BRDFPlatform._default_instance(MODEL_REGISTRY[name])
    inst === nothing || push!(models, inst)
end

println("Benchmarking $(length(models)) analytical models over $n_dirs direction pairs ...")
model_rows = evaluate_benchmark(models; n_dirs = n_dirs)
sort!(model_rows; by = r -> r.ns_per_eval)

println("\n", "="^60)
println("ANALYTICAL MODELS")
println("="^60)
@printf("%-34s %12s %12s\n", "model", "ns/eval", "Meval/s")
for r in model_rows
    @printf("%-34s %12.1f %12.2f\n", r.entry, r.ns_per_eval, r.Meval_per_s)
end

# ---------------------------------------------------------------------------
# 2. Measured formats
# ---------------------------------------------------------------------------

println("\n", "="^60)
println("MEASURED FORMATS")
println("="^60)
@printf("%-10s %-32s %10s %10s %10s\n", "format", "material", "MB", "ns/eval", "Meval/s")

fmt_rows = NamedTuple[]
for (fmt, paths) in SETS
    per = Float64[]
    for p in paths
        if !isfile(p)
            @warn "missing file, skipping" path = p
            continue
        end
        data = try
            d = load_brdf(p); d.status == :ok ? d : nothing
        catch e
            @warn "load failed, skipping" path = p exception = e; nothing
        end
        data === nothing && continue
        row = only(evaluate_benchmark(Any[data]; n_dirs = n_dirs))
        mb = filesize(p) / 1024^2
        push!(per, row.ns_per_eval)
        push!(fmt_rows, (format = fmt, material = data.name, file_mb = mb,
                         ns_per_eval = row.ns_per_eval, Meval_per_s = row.Meval_per_s))
        @printf("%-10s %-32s %10.1f %10.1f %10.2f\n", fmt, data.name, mb,
                row.ns_per_eval, row.Meval_per_s)
    end
    if !isempty(per)
        @printf("%-10s %-32s %10s %10.1f %10.2f   <- mean of %d\n",
                fmt, "MEAN", "", mean(per), 1000 / mean(per), length(per))
    end
end

# ---------------------------------------------------------------------------
# 3. Export
# ---------------------------------------------------------------------------

open(joinpath(outdir, "models.csv"), "w") do io
    println(io, "model,ns_per_eval,Meval_per_s")
    for r in model_rows
        @printf(io, "%s,%.3f,%.4f\n", r.entry, r.ns_per_eval, r.Meval_per_s)
    end
end
open(joinpath(outdir, "formats.csv"), "w") do io
    println(io, "format,material,file_mb,ns_per_eval,Meval_per_s")
    for r in fmt_rows
        @printf(io, "%s,%s,%.3f,%.3f,%.4f\n", r.format, r.material, r.file_mb,
                r.ns_per_eval, r.Meval_per_s)
    end
end

# LaTeX: one combined table, models then per-format means.
open(joinpath(outdir, "eval_timing.tex"), "w") do io
    println(io, "% Auto-generated by tools/eval_timing.jl")
    println(io, "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lrr}")
    println(io, "\\hline")
    println(io, "\\textbf{Evaluation target} & \\textbf{ns / eval} & \\textbf{Meval / s} \\\\")
    println(io, "\\hline")
    for r in model_rows
        @printf(io, "%s & %.1f & %.2f \\\\\n",
                replace(String(r.entry), "_" => "\\_"), r.ns_per_eval, r.Meval_per_s)
    end
    println(io, "\\hline")
    for (fmt, _) in SETS
        sel = [r for r in fmt_rows if r.format == fmt]
        isempty(sel) && continue
        m = mean([r.ns_per_eval for r in sel])
        @printf(io, "%s measured lookup & %.1f & %.2f \\\\\n", fmt, m, 1000 / m)
    end
    println(io, "\\hline\n\\end{tabular}")
    println(io, "\\caption[Evaluation cost per model and format]{Cost of one BRDF ",
                "evaluation. Analytical models are timed on a default instance, measured ",
                "formats as the mean over five materials (MERL, UTIA) or the full ",
                "62-material corpus (RGL). Timed with BenchmarkTools over ",
                "$(n_dirs) direction pairs.}")
    println(io, "\\label{tab:eval-timing}\n\\end{table}")
end

println("\nWrote models.csv, formats.csv, eval_timing.tex in $outdir")
