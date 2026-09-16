"""
Tests for the CPU sphere renderer (render_brdf_sphere) and the image-space
SSIM metric (image_ssim).

The CPU renderer mirrors the browser pipeline (renderer.js + universal.glsl):
color = f(wi,wo) · NdotL · intensity, Reinhard tone mapping, no gamma. The
analytic checks below derive expected pixel values from that formula.
"""

using Test
using BRDFPlatform
using Colors

@testset "Image metrics (CPU renderer + SSIM)" begin

    @testset "render_brdf_sphere: Lambertian analytic check" begin
        albedo = 0.5f0
        model = LambertianModel(albedo)
        # Light on the camera axis (θ=0): at the sphere center pixel the normal,
        # view and light all align, NdotL = 1 and (with intensity = π) the
        # pre-tone-map color is exactly the albedo → Reinhard a/(1+a).
        img, mask = render_brdf_sphere(model, 0.0, 0.0; resolution = 65)
        c = Int(cld(65, 2))
        @test mask[c, c]
        expected = albedo / (1 + albedo)
        @test img[c, c].r ≈ expected atol = 1e-3
        @test img[c, c].g ≈ expected atol = 1e-3

        # Silhouette-exterior pixels are background, not shaded.
        @test !mask[1, 1]
        @test img[1, 1].r ≈ Float32(0x20 / 255) atol = 1e-6

        # Roughly the expected fraction of pixels hit the sphere: projected
        # angular radius asin(1/4) inside a 45° fov → disk radius ≈ 0.62 of
        # the half-frame → area fraction ≈ π·0.62²/4 ≈ 0.30.
        frac = count(mask) / length(mask)
        @test 0.2 < frac < 0.4
    end

    @testset "image_ssim: identity and separation" begin
        model = LambertianModel(0.5f0)
        wrapped = BRDFPlatform._model_to_measured(model, Float32[630, 532, 465], "test")

        # A model rendered against itself is pixel-identical → SSIM = 1.
        s_same = image_ssim(wrapped, model; resolution = 96)
        @test s_same.mean ≈ 1.0 atol = 1e-9
        @test s_same.min ≈ 1.0 atol = 1e-9
        @test length(s_same.per_light) == length(BRDFPlatform.SSIM_LIGHT_DIRS)

        # A clearly different albedo must score lower, and SSIM stays in [-1, 1].
        s_diff = image_ssim(wrapped, LambertianModel(0.1f0); resolution = 96)
        @test s_diff.mean < 0.99
        @test -1.0 <= s_diff.min <= s_diff.mean <= 1.0

        # A mild albedo change should hurt less than a drastic one (monotonicity
        # sanity, not a strict SSIM property — holds for flat Lambertian images).
        s_mild = image_ssim(wrapped, LambertianModel(0.45f0); resolution = 96)
        @test s_mild.mean > s_diff.mean
    end

    @testset "reciprocity_error still available" begin
        max_abs, verdict = reciprocity_error(LambertianModel(0.5f0); n_pairs = 200)
        @test verdict === :ok
        @test max_abs ≈ 0.0 atol = 1e-6
    end

end
