#!/usr/bin/env julia
"""
tools/aggregate_sweep.jl — average a multi-material sweep bundle over materials.

`tools/fit_material.jl <folder>` writes one row per (material, run) into
`metrics_all.csv`. For a sweep whose point is "what does this factor do on
average across the dataset", that per-run table is the raw material, not the
result. This script collapses it along the material axis.

    julia --project=. tools/aggregate_sweep.jl <bundle-dir> --by optimizer
    julia --project=. tools/aggregate_sweep.jl <bundle-dir> --by samples

`--by` names the swept factor column to group on (`optimizer`, `samples`,
`loss`, `sampling`, `starts`, `cosine`).

Averages are **paired**: only materials for which *every* level of the factor
fitted successfully are counted, so the levels are compared on an identical
material set and a backend cannot look good by having failed on the hard
materials. The number of materials dropped is reported.

SSIM is the headline (image space, independent of the fitting loss). RMS-LogWE
is carried along as a secondary check. Fit time is reported as the median across
materials rather than the mean, because the first material of a run pays each
backend's one-time JIT compilation and would otherwise skew it.

Writes `<bundle-dir>/aggregate_<factor>.{md,tex,csv}`.
"""

using Printf
using Statistics

# ---------------------------------------------------------------------------
# Minimal CSV reader — the sweep tables have no quoted fields or embedded commas
# ---------------------------------------------------------------------------

function read_csv(path::String)
    lines = readlines(path)
    isempty(lines) && error("empty file: $path")
    header = String.(split(lines[1], ','))
    rows = Vector{Dict{String,String}}()
    for ln in lines[2:end]
        isempty(strip(ln)) && continue
        cells = String.(split(ln, ','))
        length(cells) == length(header) || continue
        push!(rows, Dict(zip(header, cells)))
    end
    return header, rows
end

_num(s) = isempty(strip(s)) ? NaN : parse(Float64, s)

# Sort factor levels numerically when they are numbers, else alphabetically.
function sort_levels(levels::Vector{String})
    if all(l -> tryparse(Float64, l) !== nothing, levels)
        return sort(levels; by = l -> parse(Float64, l))
    end
    return sort(levels)
end

function main(argv)
    isempty(argv) && error("usage: julia --project=. tools/aggregate_sweep.jl <bundle-dir> [--by optimizer]")
    dir = argv[1]
    factor = "optimizer"
    db = nothing
    for i in 2:length(argv)-1
        argv[i] == "--by" && (factor = argv[i+1])
        argv[i] == "--db" && (db = argv[i+1])
    end
    # Database label for the captions. Inferred from the bundle directory name
    # (bundles are named e.g. `merl_optsweep`, `rgl_sampsweep`), overridable
    # with --db for a bundle whose folder does not carry the database name.
    if db === nothing
        base = lowercase(basename(rstrip(dir, ['/', '\\'])))
        db = occursin("merl", base) ? "MERL" :
             occursin("utia", base) ? "UTIA" :
             occursin("rgl",  base) ? "RGL"  : "measured"
    end

    csv = joinpath(dir, "metrics_all.csv")
    isfile(csv) || error("no metrics_all.csv in $dir (is this a multi-material bundle?)")
    header, rows = read_csv(csv)
    factor in header || error("no column \"$factor\" in $csv. Columns: " * join(header, ", "))

    levels = sort_levels(unique(String[r[factor] for r in rows]))
    materials = unique(String[r["material"] for r in rows])

    # Paired subset: materials with an ok row for every level of the factor.
    ok = Dict{Tuple{String,String}, Dict{String,String}}()
    for r in rows
        r["status"] == "ok" || continue
        ok[(r["material"], r[factor])] = r
    end
    complete = [m for m in materials if all(l -> haskey(ok, (m, l)), levels)]
    dropped = length(materials) - length(complete)

    println("Bundle:    $dir")
    println("Factor:    $factor  ($(length(levels)) levels: ", join(levels, ", "), ")")
    println("Materials: $(length(complete)) complete", dropped > 0 ? "  ($dropped dropped as incomplete)" : "")
    println()

    # ---- aggregate ----
    stats = []
    for l in levels
        sm  = [_num(ok[(m, l)]["ssim_mean"]) for m in complete]
        smn = [_num(ok[(m, l)]["ssim_min"])  for m in complete]
        rl  = [_num(ok[(m, l)]["rmsle"])     for m in complete]
        ft  = [_num(ok[(m, l)]["fit_seconds"]) for m in complete]
        push!(stats, (level = l, ssim = mean(sm), ssim_worst = minimum(sm),
                      ssimmin = mean(smn), ssimmin_worst = minimum(smn),
                      rmsle = mean(rl), t_med = median(ft), t_mean = mean(ft)))
    end

    # Per-material win count on mean SSIM (how often each level is the best).
    wins = Dict(l => 0 for l in levels)
    for m in complete
        best = argmax([_num(ok[(m, l)]["ssim_mean"]) for l in levels])
        wins[levels[best]] += 1
    end

    # ---- console + markdown ----
    @printf("%-14s %9s %9s %9s %9s %8s %8s\n",
            factor, "SSIM", "SSIMworst", "SSIMmin", "RMS-LogWE", "t_med[s]", "wins")
    for s in stats
        @printf("%-14s %9.4f %9.4f %9.4f %9.2f %8.2f %8d\n",
                s.level, s.ssim, s.ssim_worst, s.ssimmin, 1000 * s.rmsle, s.t_med, wins[s.level])
    end

    open(joinpath(dir, "aggregate_$(factor).csv"), "w") do io
        println(io, "$factor,n_materials,ssim_mean,ssim_mean_worst,ssim_min_mean,ssim_min_worst,rmsle_mean_e3,fit_median_s,fit_mean_s,wins")
        for s in stats
            @printf(io, "%s,%d,%.6f,%.6f,%.6f,%.6f,%.4f,%.3f,%.3f,%d\n",
                    s.level, length(complete), s.ssim, s.ssim_worst, s.ssimmin,
                    s.ssimmin_worst, 1000 * s.rmsle, s.t_med, s.t_mean, wins[s.level])
        end
    end

    open(joinpath(dir, "aggregate_$(factor).md"), "w") do io
        println(io, "# Sweep over `$factor`, averaged across $(length(complete)) $db materials\n")
        dropped > 0 && println(io, "$dropped material(s) dropped (not fitted successfully at every level).\n")
        println(io, "| $factor | SSIM (mean) | SSIM (worst mat.) | SSIM\\_min (mean) | RMS-LogWE ×10⁻³ | Fit median [s] | Best on N mat. |")
        println(io, "|---|--:|--:|--:|--:|--:|--:|")
        for s in stats
            @printf(io, "| %s | %.4f | %.4f | %.4f | %.2f | %.2f | %d |\n",
                    s.level, s.ssim, s.ssim_worst, s.ssimmin, 1000 * s.rmsle, s.t_med, wins[s.level])
        end
    end

    open(joinpath(dir, "aggregate_$(factor).tex"), "w") do io
        println(io, "% Auto-generated by tools/aggregate_sweep.jl")
        println(io, "\\begin{table}[h]")
        println(io, "\\centering")
        println(io, "\\begin{tabular}{lrrrrr}")
        println(io, "\\hline")
        println(io, "\\textbf{", uppercasefirst(factor), "} & \\textbf{SSIM} & ",
                    "\\textbf{SSIM\\textsubscript{worst}} & \\textbf{SSIM\\textsubscript{min}} & ",
                    "\\textbf{RMS-LogWE} & \\textbf{Fit [s]} \\\\")
        println(io, "\\hline")
        for s in stats
            @printf(io, "%s & %.4f & %.4f & %.4f & %.2f & %.2f \\\\\n",
                    replace(s.level, "_" => "\\_"), s.ssim, s.ssim_worst,
                    s.ssimmin, 1000 * s.rmsle, s.t_med)
        end
        println(io, "\\hline")
        println(io, "\\end{tabular}")
        println(io, "\\caption[Sweep over $factor across the $db database]{Effect of \\texttt{$factor} ",
                    "on GGX fits, averaged over the $(length(complete)) $db materials fitted successfully ",
                    "at every level. SSIM is the mean over the three light directions, averaged across ",
                    "materials; SSIM\\textsubscript{worst} is the weakest single material. RMS-LogWE in ",
                    "units of \$10^{-3}\$. Fit time is the median across materials.}")
        println(io, "\\label{tab:sweep-$factor}")
        println(io, "\\end{table}")
    end

    println("\nWrote aggregate_$(factor).{csv,md,tex} in $dir")
end

main(ARGS)
