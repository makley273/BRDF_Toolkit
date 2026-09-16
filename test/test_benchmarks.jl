"""
Smoke tests for the thesis statistics suite (src/benchmarks.jl).

Runs on a **synthetic** BRDF — an analytical model wrapped in the `MeasuredBRDF`
interface — and tiny sample budgets. It verifies the plumbing (rows produced,
files written, metrics finite), not the numbers.

Deliberately free of data files: every measured database (MERL, RGL, UTIA) is
gitignored, so the suite has to pass on a fresh clone. `storage_stats` is the one
entry point that needs a real file on disk and is therefore exercised by the
format loader tests instead, which the user runs against their own databases.
"""

using Test
using BRDFPlatform

# Analytical stand-in for a measured table: same interface, no file required.
_synthetic_brdf() = BRDFPlatform._model_to_measured(
    LambertianModel(0.5), Float32[630, 532, 465], "synthetic")

@testset "Thesis statistics suite" begin

    @testset "model_storage_stats" begin
        rows = model_storage_stats()
        @test !isempty(rows)
        @test all(r -> r.n_params >= 1 && r.bytes > 0, rows)
        lamb = only(filter(r -> r.model == "LambertianModel", rows))
        @test lamb.n_params == 3   # RGB albedo
    end

    @testset "evaluate_benchmark" begin
        rows = evaluate_benchmark([LambertianModel(0.5), _synthetic_brdf()];
                                  n_dirs = 256)
        @test length(rows) == 2
        @test all(r -> isfinite(r.ns_per_eval) && r.ns_per_eval > 0, rows)
    end

    @testset "fit_benchmark" begin
        data = _synthetic_brdf()
        rows = fit_benchmark([data], [LambertianModel], [LevenbergMarquardt()];
                             ssim = false, metric_n = 256, time_fit = false)
        @test length(rows) == 1
        @test rows[1].status == :ok
        @test isfinite(rows[1].rmse)
        @test isfinite(rows[1].rms_crwe)
    end

    @testset "write_stats" begin
        rows = [(name = "a_b", value = 1.23456), (name = "c", value = 2.0)]
        base = joinpath(mktempdir(), "table")
        written = write_stats(rows, base)
        @test length(written) == 3
        @test all(isfile, written)
        csv = read(base * ".csv", String)
        @test occursin("name,value", csv)
        @test occursin("1.235", csv)          # 4 significant digits
        tex = read(base * ".tex", String)
        @test occursin("\\toprule", tex)
        @test occursin("a\\_b", tex)          # underscore escaped for LaTeX
    end
end
