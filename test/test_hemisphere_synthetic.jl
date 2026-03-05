"""
Test hemisphere plot with synthetic BRDF data.

Creates simple synthetic BRDFs with known properties to verify visualization works correctly.
"""

using BRDFPlatform
using Test
using WGLMakie: Figure  # Import Figure type for testing

println("="^70)
println("Testing Hemisphere Plot with Synthetic Data")
println("="^70)

@testset "Synthetic BRDF Visualization" begin
    # MERL dimensions
    n_theta_h = 90
    n_theta_d = 90
    n_phi_d = 180
    n_samples = n_theta_h * n_theta_d * n_phi_d

    # Helper: create a MeasuredBRDF from a flat samples matrix using MERLData
    function make_synthetic_brdf(samples_matrix::Matrix{Float32}, name::String)
        merl_data = BRDFPlatform.MERLData(samples_matrix, n_theta_h, n_theta_d, n_phi_d)
        wavelengths = Float32[630, 532, 465]
        return MeasuredBRDF(
            merl_data, wavelengths, name,
            source_format=:synthetic, status=:ok, error_msg=nothing
        )
    end

    @testset "Uniform Lambertian (Gray)" begin
        println("\n📊 Creating synthetic Lambertian BRDF (constant gray)...")

        # Create constant albedo = 0.3 (30% reflectance)
        albedo = 0.3f0
        samples = fill(albedo, 3, n_samples)

        brdf = make_synthetic_brdf(samples, "synthetic_lambertian_gray")

        # Create plot
        fig = create_simple_hemisphere_plot(brdf)

        @test fig !== nothing
        @test fig isa Figure

        println("✅ Lambertian gray hemisphere plot created")
        println("   Expected: Roughly spherical lobe (cosine weighted)")
    end

    @testset "Colored Lambertian (Red)" begin
        println("\n📊 Creating synthetic red Lambertian BRDF...")

        # Red material: high R, low G and B
        samples = zeros(Float32, 3, n_samples)
        samples[1, :] .= 0.8f0  # R
        samples[2, :] .= 0.1f0  # G
        samples[3, :] .= 0.1f0  # B

        brdf = make_synthetic_brdf(samples, "synthetic_lambertian_red")

        fig = create_simple_hemisphere_plot(brdf)

        @test fig !== nothing

        println("✅ Red Lambertian hemisphere plot created")
        println("   Expected: Red-colored spherical lobe")
    end

    @testset "Black Material (Absorbing)" begin
        println("\n📊 Creating synthetic black (absorbing) material...")

        # All zeros = perfect absorber
        samples = zeros(Float32, 3, n_samples)

        brdf = make_synthetic_brdf(samples, "synthetic_black")

        fig = create_simple_hemisphere_plot(brdf)

        @test fig !== nothing

        println("✅ Black material hemisphere plot created")
        println("   Expected: Very flat surface (no reflection)")
    end

    @testset "Bright Material" begin
        println("\n📊 Creating synthetic bright material...")

        # High albedo (close to white)
        samples = fill(Float32(0.9), 3, n_samples)

        brdf = make_synthetic_brdf(samples, "synthetic_bright")

        fig = create_simple_hemisphere_plot(brdf)

        @test fig !== nothing

        println("✅ Bright material hemisphere plot created")
        println("   Expected: Large, bright lobe")
    end

    @testset "Different Incident Angles" begin
        println("\n📊 Testing different incident light directions...")

        # Use gray Lambertian
        samples = fill(Float32(0.5), 3, n_samples)

        brdf = make_synthetic_brdf(samples, "synthetic_varying_angles")

        # Test various incident angles
        test_angles = [
            (0.0f0, 0.0f0, "Normal incidence"),
            (Float32(π/6), 0.0f0, "30° from normal"),
            (Float32(π/4), 0.0f0, "45° from normal"),
            (Float32(π/3), 0.0f0, "60° from normal"),
        ]

        for (wi_theta, wi_phi, desc) in test_angles
            fig = create_hemisphere_plot(brdf, wi_theta, wi_phi, resolution=30)
            @test fig !== nothing
            println("  ✓ $desc - plot created")
        end

        println("✅ All incident angles tested successfully")
    end
end

println("\n" * "="^70)
println("✅ All synthetic BRDF visualization tests passed!")
println("="^70)
println("\nThese tests verify:")
println("  ✓ Hemisphere plots can be created without errors")
println("  ✓ Different materials (gray, colored, black, bright) render")
println("  ✓ Different incident angles work correctly")
println("\nFor visual validation with real data:")
println("  julia --project=. test/test_hemisphere_visual.jl path/to/file.binary")
println("="^70)
