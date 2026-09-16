#!/usr/bin/env julia
"""
tools/neural_compare.jl — compare GGX against the neural model across a database.

    julia --project=. tools/neural_compare.jl <bundle-dir> [<bundle-dir> ...]

Each bundle is a multi-material `tools/fit_material.jl` run holding GGX and
neural rows under both sampling strategies. Following the method of
\\autoref{sec:results-setup}, the better of the two samplings is taken per
material, and because the selection metric would otherwise bias the comparison,
each metric selects its own best row: SSIM is reported from the sampling that
maximises SSIM, RMS-LogWE from the sampling that minimises RMS-LogWE.

Variants are GGX, neural in isotropic input mode, and neural in anisotropic
input mode, the last only where the bundle contains it.
"""

using Printf, Statistics

function load(path)
    lines = readlines(path)
    h = split(lines[1], ",")
    idx = Dict(String(h[i]) => i for i in eachindex(h))
    rows = [split(l, ",") for l in lines[2:end] if !isempty(strip(l))]
    filter!(r -> length(r) == length(h) && r[idx["status"]] == "ok", rows)
    return idx, rows
end

variant(r, idx) = r[idx["optimizer"]] == "neural" ?
    (r[idx["aniso"]] == "true" ? "Neural (aniso)" : "Neural (iso)") :
    String(r[idx["model"]])

const ORDER = ["GGXModel", "GGXAnisotropicModel", "Neural (iso)", "Neural (aniso)"]
const LABEL = Dict("GGXModel" => "GGX", "GGXAnisotropicModel" => "GGX aniso",
                   "Neural (iso)" => "Neural (iso)", "Neural (aniso)" => "Neural (aniso)")

# Variants to leave out entirely, e.g. `--drop GGXModel` when the anisotropic
# model is used as the single analytical baseline (on isotropic materials the
# two coincide, so carrying both only widens the table).
const DROP = let i = findfirst(==("--drop"), ARGS)
    i === nothing || i == length(ARGS) ? String[] : String.(split(ARGS[i+1], ','))
end

# The analytical baseline every other variant is compared against: the most
# general analytical model actually present, after DROP is applied.
function baseline(vars)
    for v in ("GGXAnisotropicModel", "GGXModel")
        v in vars && return v
    end
    return first(vars)
end

function report(label, paths)
    idx = Dict{String,Int}()
    rows = Vector{Vector{SubString{String}}}()
    for p in paths
        f = joinpath(p, "metrics_all.csv")
        isfile(f) || (@warn "no metrics_all.csv" dir = p; continue)
        i, r = load(f)
        isempty(idx) && (idx = i)
        append!(rows, r)
    end
    isempty(rows) && return
    mats = unique([String(r[idx["material"]]) for r in rows])
    vars = [v for v in ORDER if v ∉ DROP && any(r -> variant(r, idx) == v, rows)]
    isempty(vars) && return
    base = baseline(vars)
    num(r, k) = parse(Float64, r[idx[k]])

    println("\n", "="^78)
    println(label, "   ", length(mats), " materials")
    println("="^78)
    @printf("%-16s %8s %9s %10s %10s %9s %8s\n",
            "model", "SSIM", "SSIM_min", "SSIM_worst", "RMS-LogWE", "params", "fit[s]")

    best = Dict{Tuple{String,String}, NamedTuple}()
    for m in mats, v in vars
        sel = [r for r in rows if String(r[idx["material"]]) == m && variant(r, idx) == v]
        isempty(sel) && continue
        bs = sel[argmax([num(r, "ssim_mean") for r in sel])]
        be = sel[argmin([num(r, "rmsle") for r in sel])]
        best[(m, v)] = (ssim = num(bs, "ssim_mean"), ssim_min = num(bs, "ssim_min"),
                        rmsle = num(be, "rmsle"), params = num(be, "n_params"),
                        secs = num(be, "fit_seconds"))
    end

    complete = [m for m in mats if all(v -> haskey(best, (m, v)), vars)]
    for v in vars
        s  = [best[(m, v)].ssim     for m in complete]
        sm = [best[(m, v)].ssim_min for m in complete]
        e  = [best[(m, v)].rmsle    for m in complete]
        p  = [best[(m, v)].params   for m in complete]
        t  = [best[(m, v)].secs     for m in complete]
        @printf("%-16s %8.4f %9.4f %10.4f %10.2f %9d %8.2f\n",
                LABEL[v], mean(s), mean(sm), minimum(s), 1000 * mean(e),
                round(Int, median(p)), median(t))
    end

    # Pairwise: how often does each variant beat the analytical baseline?
    for v in vars
        v == base && continue
        w = count(m -> best[(m, v)].ssim > best[(m, base)].ssim, complete)
        d = [best[(m, v)].ssim - best[(m, base)].ssim for m in complete]
        @printf("  %s beats %s on %d/%d materials (mean ΔSSIM %+.4f, worst %+.4f, best %+.4f)\n",
                LABEL[v], LABEL[base], w, length(complete), mean(d), minimum(d), maximum(d))
    end
    if "Neural (aniso)" in vars && "Neural (iso)" in vars
        w = count(m -> best[(m, "Neural (aniso)")].ssim > best[(m, "Neural (iso)")].ssim, complete)
        d = [best[(m, "Neural (aniso)")].ssim - best[(m, "Neural (iso)")].ssim for m in complete]
        @printf("  anisotropic input mode helps on %d/%d materials (mean ΔSSIM %+.4f, best %+.4f)\n",
                w, length(complete), mean(d), maximum(d))
    end
    return complete, vars, best
end

# Each argument is either a bundle directory, or "Label=dir1,dir2,..." to merge
# several bundles of the same database (e.g. the neural run and the GGX
# anisotropic run, which were fitted separately but describe the same materials).
_skip = let i = findfirst(==("--drop"), ARGS)
    i === nothing ? Int[] : [i, i + 1]
end
for (n, a) in enumerate(ARGS)
    n in _skip && continue
    if occursin('=', a)
        label, dirs = split(a, '=', limit = 2)
        report(String(label), String.(split(dirs, ',')))
    else
        report(basename(rstrip(a, ['/', '\\'])), [a])
    end
end
