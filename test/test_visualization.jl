"""
Test visualization components without actual MERL data.
"""

using BRDFPlatform
using Test

println("=" ^ 70)
println("Testing Visualization Components")
println("=" ^ 70)

@testset "Visualization Tests" begin
    @testset "Module Loading" begin
        @test isdefined(BRDFPlatform, :create_hemisphere_plot)
        @test isdefined(BRDFPlatform, :create_simple_hemisphere_plot)
        @test isdefined(BRDFPlatform, :hemisphere_coords)
        @test isdefined(BRDFPlatform, :evaluate_merl_at_direction)
        println("✅ All visualization functions defined")
    end

    @testset "Hemisphere Coordinates" begin
        # Test zenith (theta=0)
        coords = BRDFPlatform.hemisphere_coords(0.0, 0.0)
        @test coords[1] ≈ 0.0 atol=1e-6
        @test coords[2] ≈ 0.0 atol=1e-6
        @test coords[3] ≈ 1.0 atol=1e-6

        # Test horizon (theta=π/2)
        coords = BRDFPlatform.hemisphere_coords(π/2, 0.0)
        @test coords[1] ≈ 1.0 atol=1e-6
        @test coords[2] ≈ 0.0 atol=1e-6
        @test coords[3] ≈ 0.0 atol=1e-6

        # Test 45 degrees
        coords = BRDFPlatform.hemisphere_coords(π/4, 0.0)
        @test coords[1] ≈ sin(π/4) atol=1e-6
        @test coords[3] ≈ cos(π/4) atol=1e-6

        println("✅ Hemisphere coordinate conversion works")
    end

    @testset "Interactive App" begin
        @test isdefined(BRDFPlatform, :run_interactive_app)
        @test isdefined(BRDFPlatform, :create_interactive_app)
        println("✅ Interactive app functions defined")
    end
end

println("\n" * "=" ^ 70)
println("✅ Visualization tests passed!")
println("=" ^ 70)
println("\nTo test with actual BRDF data:")
println("  1. Get a MERL .binary file")
println("  2. Run: julia --project=. start_interactive.jl")
println("  3. Open http://127.0.0.1:8000")
println("  4. Upload the .binary file")
println("  5. Adjust the incident angle sliders")
println("=" ^ 70)
