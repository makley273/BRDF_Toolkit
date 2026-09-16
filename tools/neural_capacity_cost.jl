#!/usr/bin/env julia
"""
tools/neural_capacity_cost.jl — evaluation time and weight storage for every
architecture in the neural capacity sweep (`tools/neural_capacity_figure.jl`,
`sec:results-capacity` in `.thesis/latex/content/results.tex`).

    julia --project=. tools/neural_capacity_cost.jl [--n-dirs 4096] [--out exports/fits/neural_capacity_cost]

Both costs are a function of the architecture's layer shapes alone, not of the
trained weight values, so this builds one freshly-initialised (untrained)
`NeuralBRDFModel` per (hidden sizes, input mode) combination rather than
re-fitting the sweep — `evaluate` reads every weight exactly once regardless of
its value, and the on-disk size is the weight count times 4 bytes either way.
`hidden_act`/`out_act` are fixed at `tools/fit_material.jl`'s defaults
(`relu`/`softplus`), which is what the sweep itself used.

Timing reuses `evaluate_benchmark` from `src/benchmarks.jl` (BenchmarkTools
`@belapsed`, minimum over auto-tuned samples, over the same deterministic
direction lattice as every other thesis timing number), so the ns-per-eval
figures are directly comparable to \\autoref{tab:format-lookup}.
"""

using BRDFPlatform
using Random

_argval(flag, default) = (i = findfirst(==(flag), ARGS);
                          i === nothing || i == length(ARGS) ? default : ARGS[i+1])

n_dirs = parse(Int, _argval("--n-dirs", "4096"))
outdir = _argval("--out", "exports/fits/neural_capacity_cost")

const ARCHS = [[6, 6], [11, 11], [21, 21], [32, 32], [64, 64], [128, 128]]
const HIDDEN_ACT, OUT_ACT = :relu, :softplus

# A random (untrained) instance with the right layer shapes for `hidden`/aniso.
# Weight values don't affect evaluate cost or storage size, only shapes do.
function random_model(hidden::Vector{Int}, anisotropic::Bool; seed::Int)
    rng = MersenneTwister(seed)
    arch = vcat(anisotropic ? 6 : 4, hidden, 3)   # input dim, hidden layers, RGB output
    n = length(arch) - 1
    Ws = [randn(rng, Float32, arch[i+1], arch[i]) for i in 1:n]
    bs = [randn(rng, Float32, arch[i+1]) for i in 1:n]
    return NeuralBRDFModel(Ws, bs, HIDDEN_ACT, OUT_ACT, arch, anisotropic)
end

models = NamedTuple[]
for hidden in ARCHS, anisotropic in (false, true)
    m = random_model(hidden, anisotropic; seed = 1)
    push!(models, (arch = join(hidden, "x"), mode = anisotropic ? "anisotropic" : "isotropic",
                   model = m))
end

println("Benchmarking $(length(models)) architecture x mode combinations ",
        "($(n_dirs) direction pairs each) ...")

storage = model_storage_stats(models = [m.model for m in models])
timing  = evaluate_benchmark([m.model for m in models]; n_dirs = n_dirs)

# Both benchmarks return one row per entry of `models`, in the same order, so
# zip by position and fold in the architecture/mode label plus the on-disk size.
rows = NamedTuple[]
for (i, m) in enumerate(models)
    n_params = storage[i].n_params
    push!(rows, (
        arch          = m.arch,
        mode          = m.mode,
        n_params      = n_params,
        disk_KB       = n_params * sizeof(Float32) / 1024,      # weights only, single precision
        memory_bytes  = storage[i].bytes,                        # in-process Julia object size
        ns_per_eval   = timing[i].ns_per_eval,
        Meval_per_s   = timing[i].Meval_per_s,
    ))
end

for r in rows
    println("  ", rpad("$(r.arch) ($(r.mode))", 22),
            "n=", lpad(r.n_params, 6), "  ",
            "disk=", lpad(round(r.disk_KB, digits = 2), 8), " KB  ",
            "eval=", lpad(round(r.ns_per_eval, digits = 1), 8), " ns  ",
            "(", round(r.Meval_per_s, digits = 2), " Mevals/s)")
end

mkpath(outdir)
written = write_stats(rows, joinpath(outdir, "neural_capacity_cost"))
println("\nWrote ", join(written, ", "))
