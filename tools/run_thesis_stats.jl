"""
Run the full thesis statistics suite and write the tables to .thesis/stats/.

Usage:
    julia --project=. tools/run_thesis_stats.jl <file-or-dir> [more files/dirs...]

Each argument is a BRDF data file or a directory (scanned non-recursively for
files; format auto-detection decides what loads). Options via environment:

    BRDF_STATS_OUT=path     output directory (default .thesis/stats)
    BRDF_STATS_SSIM=0       skip the image-space SSIM metric (much faster)
    BRDF_STATS_LUT=0        skip the LUT generation benchmark

Example:
    julia --project=. tools/run_thesis_stats.jl test/data/bird path/to/merl/red-plastic.binary
"""

using BRDFPlatform

function collect_paths(args)
    paths = String[]
    for a in args
        if isdir(a)
            append!(paths, filter(isfile, readdir(a; join = true)))
        elseif isfile(a)
            push!(paths, a)
        else
            @warn "Not found, skipping" path = a
        end
    end
    return paths
end

paths = collect_paths(ARGS)
if isempty(paths)
    println("No input files. Usage: julia --project=. tools/run_thesis_stats.jl <file-or-dir>...")
    exit(1)
end

outdir = get(ENV, "BRDF_STATS_OUT", ".thesis/stats")
ssim   = get(ENV, "BRDF_STATS_SSIM", "1") != "0"
lut    = get(ENV, "BRDF_STATS_LUT", "1") != "0"

run_thesis_benchmarks(paths; outdir = outdir, ssim = ssim, lut = lut)
