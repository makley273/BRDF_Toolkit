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

        # Test run_app exists
        @test isdefined(BRDFPlatform, :run_app)
        @test isdefined(BRDFPlatform, :run_interactive_app)

        # Test component creation functions exist
        @test isdefined(BRDFPlatform, :create_header)
        @test isdefined(BRDFPlatform, :create_footer)
        @test isdefined(BRDFPlatform, :create_sidebar)
        @test isdefined(BRDFPlatform, :create_plugin_panel)

        # Test component creation
        header = BRDFPlatform.create_header()
        @test !isnothing(header)

        footer = BRDFPlatform.create_footer()
        @test !isnothing(footer)

        sidebar = BRDFPlatform.create_sidebar()
        @test !isnothing(sidebar)

        plugin_panel = BRDFPlatform.create_plugin_panel()
        @test !isnothing(plugin_panel)
    end

    @testset "Visualization" begin
        # Test visualization functions
        @test isdefined(BRDFPlatform, :create_hemisphere_plot)
        @test isdefined(BRDFPlatform, :hemisphere_coords)
        @test isdefined(BRDFPlatform, :evaluate_merl_at_direction)

        # Test hemisphere coordinate conversion
        coords = BRDFPlatform.hemisphere_coords(0.0, 0.0)
        @test coords[3] ≈ 1.0 atol=1e-6  # Zenith

        coords = BRDFPlatform.hemisphere_coords(π/2, 0.0)
        @test coords[1] ≈ 1.0 atol=1e-6  # Horizon
    end

end
