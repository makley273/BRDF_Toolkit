#!/usr/bin/env julia
"""
tools/neural_overview.jl — collect every neural BRDF fit on record into one table.

The neural capacity study grew across many separate `tools/fit_material.jl` runs,
one material at a time, so the results sit scattered over `exports/fits/*`. This
walks those bundles, pulls out the neural rows, and reports which architectures
were actually tried on which material.

    julia --project=. tools/neural_overview.jl [exports/fits] [-o out-dir]

Bundles that carry a merged `metrics_all.csv` are read from that file and their
per-material subdirectories are skipped, so a material is never counted twice.
Where the same (material, architecture, input mode) was fitted more than once,
the best RMS-LogWE is kept and the repeat count is reported.
"""

using Printf
using Statistics

root = isempty(ARGS) || startswith(ARGS[1], "-") ? "exports/fits" : ARGS[1]
_oi = findfirst(==("-o"), ARGS)
outdir = (_oi === nothing || _oi == length(ARGS)) ? root : ARGS[_oi + 1]

function read_csv(path)
    lines = readlines(path)
    isempty(lines) && return String[], Vector{Vector{SubString{String}}}()
    h = split(lines[1], ",")
    rows = [split(l, ",") for l in lines[2:end] if !isempty(strip(l))]
    filter!(r -> length(r) == length(h), rows)
    return h, rows
end

# Strip the trailing _<date>_<time> a bundle directory carries.
clean_name(s) = replace(s, r"_\d{8}_\d{6}$" => "")

records = Dict{Tuple{String,String,String}, NamedTuple}()   # (material, arch, aniso)
sources = Dict{Tuple{String,String,String}, Set{String}}()

for bundle in sort(readdir(root; join = true))
    isdir(bundle) || continue
    merged = joinpath(bundle, "metrics_all.csv")
    files = isfile(merged) ? [merged] :
            vcat(isfile(joinpath(bundle, "metrics.csv")) ? [joinpath(bundle, "metrics.csv")] : String[],
                 [joinpath(bundle, d, "metrics.csv") for d in readdir(bundle)
                  if isdir(joinpath(bundle, d)) && isfile(joinpath(bundle, d, "metrics.csv"))])
    for f in files
        h, rows = read_csv(f)
        isempty(h) && continue
        idx = Dict(String(h[i]) => i for i in eachindex(h))
        haskey(idx, "optimizer") || continue
        for r in rows
            r[idx["optimizer"]] == "neural" || continue
            r[idx["status"]] == "ok" || continue
            mat = haskey(idx, "material") ? String(r[idx["material"]]) :
                  clean_name(basename(dirname(f)))
            arch  = String(r[idx["arch"]])
            aniso = String(r[idx["aniso"]])
            key = (mat, arch, aniso)
            rmsle = parse(Float64, r[idx["rmsle"]])
            rec = (weights = parse(Int, r[idx["n_params"]]), rmsle = rmsle,
                   ssim = parse(Float64, r[idx["ssim_mean"]]),
                   ssim_min = parse(Float64, r[idx["ssim_min"]]),
                   secs = parse(Float64, r[idx["fit_seconds"]]))
            old = get(records, key, nothing)
            (old === nothing || rmsle < old.rmsle) && (records[key] = rec)
            push!(get!(sources, key, Set{String}()), relpath(f, root))
        end
    end
end

isempty(records) && (println("no neural fits found under $root"); exit(0))

mats  = sort(unique([k[1] for k in keys(records)]))
archs = unique([k[2] for k in keys(records)])
# Order architectures by weight count (proxy for capacity), not alphabetically.
wof(a) = minimum([records[k].weights for k in keys(records) if k[2] == a])
sort!(archs; by = wof)

println("Neural fits on record under $root")
println("materials: $(length(mats))   architectures: $(length(archs))   ",
        "(material, arch, mode) combinations: $(length(records))\n")

# --- coverage matrix: which architectures exist per material ---
println("Coverage (i = isotropic, a = anisotropic, . = not fitted)")
@printf("%-26s", "material")
for a in archs; @printf("%10s", a); end
println()
for m in mats
    @printf("%-26s", m)
    for a in archs
        i = haskey(records, (m, a, "false")); n = haskey(records, (m, a, "true"))
        @printf("%10s", i && n ? "ia" : i ? "i" : n ? "a" : ".")
    end
    println()
end

open(joinpath(outdir, "neural_overview.csv"), "w") do io
    println(io, "material,arch,anisotropic,weights,rmsle,ssim_mean,ssim_min,fit_seconds,n_sources")
    for m in mats, a in archs, an in ("false", "true")
        haskey(records, (m, a, an)) || continue
        r = records[(m, a, an)]
        @printf(io, "%s,%s,%s,%d,%.6f,%.6f,%.6f,%.3f,%d\n",
                m, a, an, r.weights, r.rmsle, r.ssim, r.ssim_min, r.secs,
                length(sources[(m, a, an)]))
    end
end

open(joinpath(outdir, "neural_overview.md"), "w") do io
    println(io, "# Neural fits on record\n")
    println(io, "$(length(mats)) material(s), $(length(archs)) architecture(s), ",
                "$(length(records)) (material, architecture, input mode) combinations.\n")
    println(io, "| Material | Arch | Mode | Weights | RMS-LogWE ×10⁻³ | SSIM | SSIM min | Fit [s] |")
    println(io, "|---|---|---|--:|--:|--:|--:|--:|")
    for m in mats, a in archs, an in ("false", "true")
        haskey(records, (m, a, an)) || continue
        r = records[(m, a, an)]
        @printf(io, "| %s | %s | %s | %d | %.2f | %.4f | %.4f | %.1f |\n",
                m, a, an == "true" ? "aniso" : "iso", r.weights,
                1000 * r.rmsle, r.ssim, r.ssim_min, r.secs)
    end
end

println("\nWrote neural_overview.{csv,md} in $outdir")

# Which materials carry a real capacity sweep (more than two architectures)?
println("\nMaterials with a capacity sweep (>2 architectures in one input mode):")
for m in mats
    for an in ("false", "true")
        n = count(a -> haskey(records, (m, a, an)), archs)
        n > 2 && @printf("  %-26s %-6s %d architectures\n", m, an == "true" ? "aniso" : "iso", n)
    end
end
