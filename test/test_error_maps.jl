"""
Tests for the ErrorMap implementations, focusing on the signed maps added for
over/under-estimation feedback (SignedLogDifference, SMAPEHeat).
"""

using Test
using BRDFPlatform
using Colors

# Gray helper: luminance-only colors so per-channel and luminance maps agree.
_gray(v) = RGB{Float32}(v, v, v)

@testset "Error maps" begin

    @testset "SignedLogDifference: sign and saturation" begin
        em = SignedLogDifference()  # max_logdiff = 2.0

        # Perfect match → white (the diverging map's neutral point).
        c = evaluate_error(em, _gray(0.25f0), _gray(0.25f0))
        @test c.r ≈ 1.0f0 atol = 1e-5
        @test c.g ≈ 1.0f0 atol = 1e-5
        @test c.b ≈ 1.0f0 atol = 1e-5

        # Model brighter than measured → red side (r stays 1, g/b drop).
        c = evaluate_error(em, _gray(0.1f0), _gray(0.3f0))
        @test c.r ≈ 1.0f0
        @test c.g < 1.0f0
        @test c.g ≈ c.b

        # Model darker than measured → blue side (b stays 1, r/g drop).
        c = evaluate_error(em, _gray(0.3f0), _gray(0.1f0))
        @test c.b ≈ 1.0f0
        @test c.r < 1.0f0
        @test c.r ≈ c.g

        # A factor-4 error is exactly 2 octaves → full saturation (pure red),
        # and anything beyond clamps there rather than wrapping/overflowing.
        c4 = evaluate_error(em, _gray(0.1f0), _gray(0.4f0))
        @test c4.g < 0.05f0                      # ≈ saturated (eps shifts it slightly)
        c100 = evaluate_error(em, _gray(0.01f0), _gray(1.0f0))
        @test c100.r ≈ 1.0f0
        @test c100.g ≈ 0.0f0 atol = 1e-5
        @test c100.b ≈ 0.0f0 atol = 1e-5
    end

    @testset "SMAPEHeat: bounded and symmetric" begin
        em = SMAPEHeat()

        # Identical inputs → SMAPE 0 → jet's low end (dark blue).
        c0 = evaluate_error(em, _gray(0.5f0), _gray(0.5f0))
        @test c0.r ≈ 0.0f0 atol = 1e-5
        @test c0.b > 0.0f0

        # Symmetric: swapping measured/fitted gives the identical color.
        ca = evaluate_error(em, _gray(0.1f0), _gray(0.7f0))
        cb = evaluate_error(em, _gray(0.7f0), _gray(0.1f0))
        @test ca.r ≈ cb.r && ca.g ≈ cb.g && ca.b ≈ cb.b

        # One side ~zero → SMAPE ≈ 1 → jet's high end (dark red ≈ (0.5, 0, 0)),
        # never out of range.
        c1 = evaluate_error(em, _gray(0.0f0), _gray(1.0f0))
        @test c1.r ≈ 0.5f0 atol = 0.05
        @test c1.g ≈ 0.0f0 atol = 0.05
        @test c1.b ≈ 0.0f0 atol = 1e-5
    end

    @testset "Dropdown registry" begin
        @test AVAILABLE_ERROR_MAPS[1] isa SignedLogDifference   # default selection
        @test any(em -> em isa SMAPEHeat, AVAILABLE_ERROR_MAPS)
        # Every registered map satisfies the interface.
        for em in AVAILABLE_ERROR_MAPS
            @test error_map_label(em) isa String
            c = evaluate_error(em, _gray(0.2f0), _gray(0.4f0))
            @test all(0.0f0 .<= (c.r, c.g, c.b) .<= 1.0f0)
        end
    end

    @testset "Legends" begin
        # Every registered map provides a legend with CSS stops + three labels.
        for em in AVAILABLE_ERROR_MAPS
            spec = error_map_legend(em)
            @test spec !== nothing
            @test length(spec.colors) >= 3
            @test all(startswith.(spec.colors, "rgb("))
            @test spec.left isa String && spec.mid isa String && spec.right isa String
        end

        # SignedLogDifference legend: blue → white → red, factor from max_logdiff.
        spec = error_map_legend(SignedLogDifference())
        @test spec.colors[1] == "rgb(0,0,255)"
        @test spec.colors[(end + 1) ÷ 2] == "rgb(255,255,255)"
        @test spec.colors[end] == "rgb(255,0,0)"
        @test occursin("4", spec.right)      # max_logdiff=2 → factor 4

        # A custom ErrorMap without a legend method falls back to nothing.
        struct _NoLegendMap <: BRDFPlatform.ErrorMap end
        @test error_map_legend(_NoLegendMap()) === nothing
    end

end
