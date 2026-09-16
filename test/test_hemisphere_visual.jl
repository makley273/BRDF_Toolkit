"""
Visual validation test for hemisphere plots.

Run this script with a MERL .binary file to visually inspect the hemisphere plot.
This helps verify:
1. Coordinate transformations are correct
2. BRDF lookup works properly
3. Visualization scaling is reasonable
4. Colors match expected material appearance

Usage:
    julia --project=. test/test_hemisphere_visual.jl path/to/file.binary
"""

using BRDFPlatform
using GLMakie  # Use GLMakie for interactive desktop viewing

function main(args)
    if length(args) < 1
        println("Usage: julia --project=. test/test_hemisphere_visual.jl <path-to-merl-file.binary>")
        println("\nExpected behaviors to check visually:")
        println("  - Diffuse materials: Should show roughly spherical lobes")
        println("  - Specular materials: Should show sharp peaks near mirror direction")
        println("  - Metallic materials: Should show colored reflections")
        println("  - Grazing angles: Should show elongated lobes")
        return
    end

    filepath = args[1]

    if !isfile(filepath)
        println("Error: File not found: $filepath")
        return
    end

    println("Loading BRDF from: $filepath")
    brdf = load_brdf(BRDFPlatform.MerlFormat, filepath)

    if brdf.status != :ok
        println("Error loading BRDF: $(brdf.error_msg)")
        return
    end

    println("✅ Loaded: $(brdf.name)")
    println("   Format: $(brdf.source_format)")
    println("   Wavelengths: $(brdf.wavelengths) nm")
    println("   Samples shape: $(size(brdf.samples))")

    # Test several incident angles
    test_cases = [
        (0.0f0, 0.0f0, "Normal incidence (straight down)"),
        (Float32(π/4), 0.0f0, "45° from normal"),
        (Float32(π/3), 0.0f0, "60° from normal (grazing)"),
        (Float32(π/4), Float32(π/2), "45° from different azimuth")
    ]

    for (theta, phi, description) in test_cases
        println("\n" * "="^70)
        println("Creating hemisphere plot: $description")
        println("  wi_theta = $(rad2deg(theta))°, wi_phi = $(rad2deg(phi))°")
        println("="^70)

        try
            fig = create_hemisphere_plot(brdf, theta, phi, resolution=50)

            # Display the figure
            display(fig)

            println("\n✅ Plot created successfully!")
            println("\nVisual checks to perform:")
            println("  1. Red arrow points in correct direction?")
            println("  2. Hemisphere surface is smooth?")
            println("  3. Colors look reasonable (no extreme values)?")
            println("  4. Peak reflection near mirror direction for specular?")
            println("  5. Symmetry matches material properties?")

            println("\nPress Enter to continue to next test case...")
            readline()

        catch e
            println("❌ Error creating plot: $e")
            println(stacktrace(catch_backtrace()))
        end
    end

    println("\n" * "="^70)
    println("Visual validation complete!")
    println("="^70)
end

main(ARGS)
