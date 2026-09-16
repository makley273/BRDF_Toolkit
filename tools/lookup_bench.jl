#!/usr/bin/env julia
"""
tools/lookup_bench.jl — evaluation cost of every representation, side by side.

    julia --project=. tools/lookup_bench.jl

Times a single `lookup(entry, wi, wo)` for the three measured formats and for
the fitted models, over a fixed direction lattice. Evaluation cost depends on
the representation and not on the fitted values, so the models are trained only
briefly here: the numbers would be identical for a fully converged fit.

Also reports what the measured entry costs in memory and what the universal
4D LUT costs to generate, because those are the two numbers the evaluation time
has to be read against (see the note printed at the end).
"""

using BRDFPlatform
using Printf

const MATERIALS = [
    ("MERL", "test/data/setup/blue-metallic-paint2.binary"),
    ("UTIA", "test/data/setup/m071_fabric106.bin"),
    ("RGL",  "test/data/setup/acrylic_felt_green_rgb.bsdf"),
]

datas = Pair{String, MeasuredBRDF}[]
for (label, path) in MATERIALS
    d = load_brdf(path)
    d.status == :ok || (@warn "skipping $label ($(d.status))"; continue)
    push!(datas, label => d)
end

# Fit the models on one material. Architecture, not fit quality, sets eval cost.
ref = last(datas[1])
println("fitting reference models on $(ref.name) ...")
ggxa = fit_brdf(LBFGSB(; loss = :log, sampling = :uniform_cosine, n_samples = 1024),
                GGXAnisotropicModel, ref)
nn_i = fit_brdf(NeuralBRDFTrainer(; hidden = [21, 21], anisotropic = false,
                                    epochs = 5, n_samples = 1024), NeuralBRDFModel, ref)
nn_a = fit_brdf(NeuralBRDFTrainer(; hidden = [21, 21], anisotropic = true,
                                    epochs = 5, n_samples = 1024), NeuralBRDFModel, ref)

entries = Any[]
labels  = String[]
for (label, d) in datas
    push!(entries, d); push!(labels, "measured $label")
end
push!(entries, ggxa); push!(labels, "GGX anisotropic")
push!(entries, nn_i); push!(labels, "Neural isotropic (633 w)")
push!(entries, nn_a); push!(labels, "Neural anisotropic (675 w)")

println("\nbenchmarking lookup ...")
rows = evaluate_benchmark(entries; n_dirs = 4096)

println("\n", "="^62)
@printf("%-28s %14s %14s\n", "representation", "ns / lookup", "M lookups/s")
println("="^62)
for (l, r) in zip(labels, rows)
    @printf("%-28s %14.1f %14.2f\n", l, r.ns_per_eval, r.Meval_per_s)
end

# In-memory size of each measured dataset, for the storage/time trade-off.
println("\n", "="^62)
@printf("%-28s %14s\n", "measured dataset", "in memory [MB]")
println("="^62)
for (label, d) in datas
    @printf("%-28s %14.2f\n", "$label ($(d.name))", Base.summarysize(d) / 1024^2)
end

# LUT generation: the one-off cost that every model pays before rendering.
println("\n", "="^62)
println("universal 4D LUT (the actual render path)")
println("="^62)
for (nm, m) in [("GGX anisotropic", ggxa), ("Neural isotropic", nn_i),
                ("Neural anisotropic", nn_a), ("measured $(first(datas[1]))", ref)]
    t = @elapsed generate_lut(m; dims = (32, 32, 32, 1), dtype = :float16)
    @printf("  %-28s generate 32x32x32 f16: %6.2f s\n", nm, t)
end

println("""

Note. At render time none of the per-lookup numbers above apply: the platform
bakes every representation into the same universal 4D LUT and the shader does a
texture fetch, so rendering cost is identical for all models and independent of
the one that produced it. The lookup times matter for fitting (the optimizer
calls `lookup` on the measured data once per sample per iteration) and for LUT
generation, not for the interactive frame rate.
""")
