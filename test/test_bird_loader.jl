"""
Test BiRD JSON BRDF loader against the canonical example file.

Reference example: BiRD_view/Test BRDF data files/example.brdf (N=8).
A copy is committed to test/data/bird/example.brdf so the test does not
depend on the gitignored BiRD_view tree.
"""

using Test
using JSON3
using StaticArrays

push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using BRDFPlatform

const EXAMPLE_BRDF = joinpath(@__DIR__, "data", "bird", "example.brdf")

@assert isfile(EXAMPLE_BRDF) "Missing test fixture: $EXAMPLE_BRDF"

@testset "BiRDFormat" begin

    @testset "Registration" begin
        plugins = BRDFPlatform.list_plugins()
        @test :birdformat in plugins[:formats]
    end

    @testset "Detection" begin
        # Confidence for a real BiRD file
        c = BRDFPlatform.detect_format(BRDFPlatform.BiRDFormat, EXAMPLE_BRDF)
        @test c >= 0.6

        # Malformed JSON returns 0 (no throw)
        tmp = tempname() * ".bird"
        write(tmp, "not valid json {{{")
        try
            @test BRDFPlatform.detect_format(BRDFPlatform.BiRDFormat, tmp) == 0.0
        finally
            rm(tmp, force=true)
        end

        # Missing file returns 0
        @test BRDFPlatform.detect_format(BRDFPlatform.BiRDFormat,
                                         tempname() * ".bird") == 0.0
    end

    @testset "Parsing example.brdf" begin
        b = BRDFPlatform.load_brdf(BRDFPlatform.BiRDFormat, EXAMPLE_BRDF)
        @test b.status == :ok
        @test b.source_format == :bird
        @test BRDFPlatform.n_samples_str(b._data) == "8"

        d = b._data
        @test d isa BRDFPlatform.BiRDData
        @test length(d.theta_i) == 8
        @test length(d.phi_i)   == 8
        @test length(d.theta_r) == 8
        @test length(d.phi_r)   == 8
        @test length(d.brdf)    == 8

        # All theta_i / phi_i are 0 in the example
        @test all(d.theta_i .== 0.0f0)
        @test all(d.phi_i   .== 0.0f0)
        # theta_r = 10°, phi_r = 60° → radians
        @test all(isapprox.(d.theta_r, deg2rad(10.0f0); atol=1e-6))
        @test all(isapprox.(d.phi_r,   deg2rad(60.0f0); atol=1e-6))

        # First and last BRDF samples from spec
        @test d.brdf[1]  ≈ 0.254f0 atol=1e-6
        @test d.brdf[end] ≈ 0.301f0 atol=1e-6

        # No wavelength_r in example.brdf → default 550 nm; wavelength_i kept raw
        @test b.wavelengths == Float32[550]
        @test d.wavelength_r === nothing
        @test d.wavelength_i_raw !== nothing

        # Other optional fields stored raw
        @test d.polarization_i_raw !== nothing
        @test haskey(d.adhoc_raw, "sample_width")
    end

    @testset "Exact-direction lookup" begin
        b = BRDFPlatform.load_brdf(BRDFPlatform.BiRDFormat, EXAMPLE_BRDF)

        # wi: θi = 0 → straight up
        wi = BRDFPlatform.vec3(0.0, 0.0, 1.0)
        # wo: θr = 10°, φr = 60°
        θ = deg2rad(10.0); φ = deg2rad(60.0)
        wo = BRDFPlatform.vec3(sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ))

        rgb = BRDFPlatform.brdf_value(b, wi, wo)
        @test rgb.r == rgb.g == rgb.b              # grayscale broadcast
        @test 0.254f0 - 1e-3 <= rgb.r <= 0.301f0 + 1e-3
    end

    @testset "Smooth interpolation off-axis" begin
        b = BRDFPlatform.load_brdf(BRDFPlatform.BiRDFormat, EXAMPLE_BRDF)
        wi = BRDFPlatform.vec3(0.0, 0.0, 1.0)
        θ = deg2rad(20.0); φ = deg2rad(45.0)
        wo = BRDFPlatform.vec3(sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ))

        rgb = BRDFPlatform.brdf_value(b, wi, wo)
        @test isfinite(rgb.r)
        @test rgb.r >= 0.0f0
        # IDW is bounded by min/max of input values
        @test 0.254f0 - 1e-3 <= rgb.r <= 0.301f0 + 1e-3
    end

    @testset "Auto-detect via high-level load_brdf" begin
        b = BRDFPlatform.load_brdf(EXAMPLE_BRDF)
        @test b.status == :ok
        @test b.source_format == :bird
    end

    @testset "Length-mismatch rejection" begin
        bad = Dict(
            "metadata" => Dict("type" => "BRDF"),
            "data" => Dict(
                "theta_i" => Dict("unit" => "deg", "values" => [0, 0]),
                "phi_i"   => Dict("unit" => "deg", "values" => [0, 0]),
                "theta_r" => Dict("unit" => "deg", "values" => [10, 20]),
                "phi_r"   => Dict("unit" => "deg", "values" => [0, 0, 0]),  # wrong length
                "BRDF"    => Dict("unit" => "sr^-1", "values" => [0.1, 0.2]),
            ),
        )
        tmp = tempname() * ".bird"
        write(tmp, JSON3.write(bad))
        try
            b = BRDFPlatform.load_brdf(BRDFPlatform.BiRDFormat, tmp)
            @test b.status == :failed
            @test b.error_msg !== nothing
            @test occursin("mismatch", lowercase(b.error_msg))
        finally
            rm(tmp, force=true)
        end
    end

end
