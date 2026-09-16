"""
Test RGL EPFL .bsdf loader.

Uses the RGB sample in `test/data/RGL/`. Reference values below were produced
by the official C++ loader (powitacq_rgb, https://github.com/rgl-epfl/brdf-loader)
compiled with g++ and evaluated on the same file — the Julia port must agree
to well under 1e-3 relative error.
"""

using Test
using StaticArrays

push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using BRDFPlatform

const RGL_DATA_DIR = joinpath(@__DIR__, "data", "RGL")
const RGL_SAMPLE = joinpath(RGL_DATA_DIR, "acrylic_felt_green_rgb.bsdf")

@testset "RGLFormat" begin

    @testset "Registration" begin
        plugins = BRDFPlatform.list_plugins()
        @test :rglformat in plugins[:formats]
    end

    @testset "Detection — synthetic files" begin
        # Not a tensor file → 0
        tmp1 = tempname() * ".bsdf"
        write(tmp1, "definitely not a tensor_file, but long enough")
        try
            @test BRDFPlatform.detect_format(BRDFPlatform.RGLFormat, tmp1) == 0.0
        finally
            rm(tmp1, force=true)
        end

        # Correct magic, wrong extension → 0.9
        tmp2 = tempname() * ".dat"
        write(tmp2, vcat(Vector{UInt8}(b"tensor_file\0"), UInt8[1, 0, 0, 0, 0, 0]))
        try
            @test BRDFPlatform.detect_format(BRDFPlatform.RGLFormat, tmp2) ≈ 0.9
        finally
            rm(tmp2, force=true)
        end

        # Missing file → 0
        @test BRDFPlatform.detect_format(BRDFPlatform.RGLFormat, "no_such_file.bsdf") == 0.0
    end

    @testset "Failed load returns :failed status" begin
        brdf = BRDFPlatform.load_brdf(BRDFPlatform.RGLFormat, "no_such_file.bsdf")
        @test brdf.status == :failed
        @test brdf.error_msg !== nothing
    end

    if isfile(RGL_SAMPLE)
        @testset "Load sample file" begin
            @test BRDFPlatform.detect_format(BRDFPlatform.RGLFormat, RGL_SAMPLE) ≈ 1.0

            brdf = BRDFPlatform.load_brdf(BRDFPlatform.RGLFormat, RGL_SAMPLE)
            @test brdf.status == :ok
            @test brdf.source_format == :rgl
            @test brdf.name == "acrylic_felt_green_rgb"
            @test brdf._data.isotropic
            @test brdf._data.description == "Green acrylic felt (Edukit A4)"
            @test occursin("θᵢ", BRDFPlatform.n_samples_str(brdf._data))
        end

        @testset "Eval matches official C++ loader" begin
            brdf = BRDFPlatform.load_brdf(BRDFPlatform.RGLFormat, RGL_SAMPLE)

            # (wi, wo, fr) triples dumped by the reference implementation.
            # NOTE: the reference eval returns f_r * cos(theta_o) (Mitsuba
            # convention); our loader returns the plain BRDF, so we multiply
            # by wo.z before comparing.
            refs = [
                ((0.142066538f0, 0.0148245264f0, 0.989746094f0),
                 (-0.487714201f0, -0.552531183f0, 0.675902486f0),
                 (0.0213497449, 0.0461031757, 0.00611046143)),
                ((0.725542188f0, 0.61936909f0, 0.299950868f0),
                 (-0.671949863f0, -0.210648373f0, 0.710007489f0),
                 (0.070052132, 0.109777056, 0.0216141846)),
                ((0.161078885f0, -0.75508517f0, 0.635531187f0),
                 (-0.00777344313f0, -0.199483991f0, 0.97987026f0),
                 (0.0258504376, 0.0599292144, 0.00701350067)),
            ]

            for (wi_t, wo_t, fr_ref) in refs
                wi = SVector{3, Float32}(wi_t)
                wo = SVector{3, Float32}(wo_t)
                got = BRDFPlatform.brdf_value(brdf, wi, wo)
                for (g, r) in zip((got.r, got.g, got.b), fr_ref)
                    @test isapprox(g * wo[3], r; rtol = 1e-3)
                end
            end
        end

        @testset "Approximate reciprocity (guards the LUT phi_d fold)" begin
            # The plain BRDF must be roughly reciprocal — if a cosine factor
            # sneaks back in (reference eval returns f_r*cos), the universal
            # LUT's phi_d reciprocity fold renders a seam along the plane of
            # incidence.
            brdf = BRDFPlatform.load_brdf(BRDFPlatform.RGLFormat, RGL_SAMPLE)

            dirs = [SVector{3, Float32}(sin(t) * cos(p), sin(t) * sin(p), cos(t))
                    for t in (0.2f0, 0.6f0, 1.0f0), p in (0.0f0, 1.5f0, 3.0f0, 4.5f0)]
            total = 0.0; n = 0
            for wi in dirs, wo in dirs
                a = BRDFPlatform.brdf_value(brdf, wi, wo).g
                c = BRDFPlatform.brdf_value(brdf, wo, wi).g
                m = max(a, c)
                m < 1e-4 && continue
                total += abs(a - c) / m
                n += 1
            end
            @test n > 0
            @test total / n < 0.10   # f_r*cos would give ~0.3 here
        end

        @testset "Hemisphere boundary behavior" begin
            brdf = BRDFPlatform.load_brdf(BRDFPlatform.RGLFormat, RGL_SAMPLE)

            below = SVector{3, Float32}(0, 0, -1)
            up = SVector{3, Float32}(0, 0, 1)
            zero_rgb = BRDFPlatform.brdf_value(brdf, below, up)
            @test zero_rgb.r == 0 && zero_rgb.g == 0 && zero_rgb.b == 0

            # Normal incidence/reflection yields a finite, non-negative value
            v = BRDFPlatform.brdf_value(brdf, up, up)
            @test isfinite(v.r) && v.r >= 0
        end
    else
        @info "RGL sample file not found, skipping file-dependent tests" RGL_SAMPLE
    end
end
