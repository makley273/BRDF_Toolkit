"""
Test UTIA binary BRDF loader.

Drop a real UTIA `.bin` file into `test/data/utia/` to enable the
file-dependent tests. Tests that don't need a real file always run.
"""

using Test
using StaticArrays

push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using BRDFPlatform

const UTIA_DATA_DIR = joinpath(@__DIR__, "data", "utia")

# Pick the first .bin in the data dir, if any
function _find_utia_sample()
    isdir(UTIA_DATA_DIR) || return nothing
    for f in readdir(UTIA_DATA_DIR; join=true)
        endswith(lowercase(f), ".bin") || continue
        filesize(f) == BRDFPlatform.UTIA_EXPECTED_SIZE && return f
    end
    return nothing
end

const UTIA_SAMPLE = _find_utia_sample()

@testset "UTIAFormat" begin

    @testset "Registration" begin
        plugins = BRDFPlatform.list_plugins()
        @test :utiaformat in plugins[:formats]
    end

    @testset "Constants" begin
        @test BRDFPlatform.UTIA_EXPECTED_SIZE == 1_990_656
        @test BRDFPlatform.UTIA_NPI == 48
        @test BRDFPlatform.UTIA_NPV == 48
        @test BRDFPlatform.UTIA_STRIDE_PLANES == 6 * 6 * 48 * 48
    end

    @testset "Detection — synthetic files" begin
        # Wrong size, wrong extension → 0
        tmp1 = tempname() * ".dat"
        write(tmp1, "x")
        try
            @test BRDFPlatform.detect_format(BRDFPlatform.UTIAFormat, tmp1) == 0.0
        finally
            rm(tmp1, force=true)
        end

        # Right extension, wrong size → 0.3
        tmp2 = tempname() * ".bin"
        write(tmp2, "x")
        try
            @test BRDFPlatform.detect_format(BRDFPlatform.UTIAFormat, tmp2) ≈ 0.3
        finally
            rm(tmp2, force=true)
        end

        # Right size, wrong extension → 0.7
        tmp3 = tempname() * ".dat"
        open(tmp3, "w") do io
            write(io, zeros(UInt8, BRDFPlatform.UTIA_EXPECTED_SIZE))
        end
        try
            @test BRDFPlatform.detect_format(BRDFPlatform.UTIAFormat, tmp3) ≈ 0.7
        finally
            rm(tmp3, force=true)
        end

        # Missing file → 0
        @test BRDFPlatform.detect_format(BRDFPlatform.UTIAFormat,
                                         tempname() * ".bin") == 0.0
    end

    @testset "Loader — missing file" begin
        b = BRDFPlatform.load_brdf(BRDFPlatform.UTIAFormat,
                                   joinpath(UTIA_DATA_DIR, "does_not_exist.bin"))
        @test b.status == :failed
        @test b.source_format == :utia
        @test b.error_msg !== nothing
    end

    if UTIA_SAMPLE === nothing
        @info "Skipping UTIA file-dependent tests — drop a .bin into $UTIA_DATA_DIR to enable."
    else
        @testset "Detection — real file" begin
            c = BRDFPlatform.detect_format(BRDFPlatform.UTIAFormat, UTIA_SAMPLE)
            @test c >= 0.6
            @test c ≈ 1.0  # both .bin extension and exact size
        end

        @testset "Parsing real file" begin
            b = BRDFPlatform.load_brdf(BRDFPlatform.UTIAFormat, UTIA_SAMPLE)
            @test b.status == :ok
            @test b.source_format == :utia
            @test b.wavelengths == Float32[630, 532, 465]

            d = b._data
            @test d isa BRDFPlatform.UTIAData
            @test BRDFPlatform.n_samples_str(d) == string(6 * 6 * 48 * 48)
            @test length(d.samples) == 3 * BRDFPlatform.UTIA_STRIDE_PLANES
        end

        @testset "Lookup" begin
            b = BRDFPlatform.load_brdf(BRDFPlatform.UTIAFormat, UTIA_SAMPLE)

            # In-hemisphere directions
            for (θi_deg, φi_deg, θv_deg, φv_deg) in (
                (0.0, 0.0, 0.0, 0.0),
                (30.0, 45.0, 30.0, 225.0),
                (15.0, 90.0, 60.0, 180.0),
            )
                θi = deg2rad(θi_deg); φi = deg2rad(φi_deg)
                θv = deg2rad(θv_deg); φv = deg2rad(φv_deg)
                wi = BRDFPlatform.vec3(sin(θi)*cos(φi), sin(θi)*sin(φi), cos(θi))
                wo = BRDFPlatform.vec3(sin(θv)*cos(φv), sin(θv)*sin(φv), cos(θv))
                rgb = BRDFPlatform.brdf_value(b, wi, wo)
                @test isfinite(rgb.r) && isfinite(rgb.g) && isfinite(rgb.b)
                @test rgb.r >= 0.0f0 && rgb.g >= 0.0f0 && rgb.b >= 0.0f0
            end

            # Below the surface → zero
            wi_below = BRDFPlatform.vec3(0.0, 0.0, -1.0)
            wo_up    = BRDFPlatform.vec3(0.0, 0.0,  1.0)
            rgb0 = BRDFPlatform.brdf_value(b, wi_below, wo_up)
            @test rgb0.r == 0.0f0 && rgb0.g == 0.0f0 && rgb0.b == 0.0f0
        end

        @testset "Auto-detect" begin
            b = BRDFPlatform.load_brdf(UTIA_SAMPLE)
            @test b.status == :ok
            @test b.source_format == :utia
        end
    end
end
