"""
Test suite for BRDFPlatform

Run with: julia --project=. test/runtests.jl
"""

using Test
using BRDFPlatform
using StaticArrays

@testset "BRDFPlatform Tests" begin

    @testset "Module Loading" begin
        @test isdefined(BRDFPlatform, :BRDFData)
        @test isdefined(BRDFPlatform, :BRDFModel)
        @test isdefined(BRDFPlatform, :MeasuredBRDF)
    end

    @testset "Plugin Registry" begin
        plugins = BRDFPlatform.list_plugins()

        @test haskey(plugins, :formats)
        @test haskey(plugins, :models)
        @test haskey(plugins, :optimizers)

        # Check that MERL format is registered
        @test :merlformat in plugins[:formats]

        # Neural BRDF plugins register
        @test :neuralbrdfmodel in plugins[:models]
        @test :neuralbrdftrainer in plugins[:optimizers]

        println("Registered formats: ", plugins[:formats])
        println("Registered models: ", plugins[:models])
        println("Registered optimizers: ", plugins[:optimizers])
    end

    @testset "Type System" begin
        # Test vector utilities
        v = BRDFPlatform.vec3(1.0, 0.0, 0.0)
        @test v isa StaticArrays.SVector{3, Float32}
        @test v[1] ≈ 1.0f0

        # Test normalization
        v2 = BRDFPlatform.vec3(3.0, 4.0, 0.0)
        v2_norm = BRDFPlatform.normalize_vec(v2)
        @test sqrt(sum(v2_norm.^2)) ≈ 1.0f0

        # Test RGB
        c = BRDFPlatform.rgb(1.0, 0.5, 0.0)
        @test c.r ≈ 1.0f0
        @test c.g ≈ 0.5f0
    end

    @testset "MERLData Sampling" begin
        # Test MERLData stores grid dimensions
        data = BRDFPlatform.MERLData(zeros(Float32, 3, 90*90*180), 90, 90, 180)
        @test data.n_theta_h == 90
        @test data.n_theta_d == 90
        @test data.n_phi_d == 180
        @test BRDFPlatform.n_samples_str(data) == "1458000"
    end

    @testset "UI Components" begin
        using Bonito

        # Server entry points
        @test isdefined(BRDFPlatform, :run_platform)
        @test isdefined(BRDFPlatform, :run_interactive_app)
        @test isdefined(BRDFPlatform, :run_rusinkiewicz_app)
        @test isdefined(BRDFPlatform, :run_sampling_app)

        # Page builders behind the routes of run_platform
        @test isdefined(BRDFPlatform, :create_interactive_app)
        @test isdefined(BRDFPlatform, :create_rusinkiewicz_app)
        @test isdefined(BRDFPlatform, :create_sampling_app)

        # Build a page that needs no loaded data, to catch layout-time errors
        session = Session()
        @test !isnothing(BRDFPlatform.create_sampling_app(session))
    end

    @testset "Visualization" begin
        # Test visualization functions
        @test isdefined(BRDFPlatform, :create_multichannel_hemisphere_plot)
        @test isdefined(BRDFPlatform, :hemisphere_coords)
        @test isdefined(BRDFPlatform, :evaluate_brdf_at_direction)

        # Test hemisphere coordinate conversion
        coords = BRDFPlatform.hemisphere_coords(0.0, 0.0)
        @test coords[3] ≈ 1.0 atol=1e-6  # Zenith

        coords = BRDFPlatform.hemisphere_coords(π/2, 0.0)
        @test coords[1] ≈ 1.0 atol=1e-6  # Horizon
    end

    @testset "Unified lookup API" begin
        # Test that lookup is defined and works for analytical models
        @test isdefined(BRDFPlatform, :lookup)

        # Test lookup on a Lambertian model
        model = LambertianModel(0.5)
        wi = BRDFPlatform.vec3(0.0, 0.0, 1.0)
        wo = BRDFPlatform.vec3(0.0, 0.0, 1.0)
        result = lookup(model, wi, wo)
        @test result.r ≈ 0.5f0 / Float32(π) atol=1e-6
        @test result.g ≈ 0.5f0 / Float32(π) atol=1e-6

        # Test lookup with tuples
        result2 = lookup(model, (0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
        @test result2.r ≈ result.r atol=1e-6
    end

end

# Ashikhmin-Shirley anisotropic Phong model
include("test_ashikhmin_shirley.jl")

# Neural BRDF model + trainer (Lux/Adam)
include("test_neural_brdf.jl")

# CPU sphere renderer + image-space SSIM metric
include("test_image_metrics.jl")

# Error maps (signed log difference, SMAPE, dropdown registry)
include("test_error_maps.jl")

# Universal LUT half-angle sampling (over-complete Rusinkiewicz box → horizon clamp)
include("test_lut_sampling.jl")

# Thesis statistics suite (storage / timing / fit-accuracy benchmark tables)
include("test_benchmarks.jl")
