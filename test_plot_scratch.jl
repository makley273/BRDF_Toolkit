#!/usr/bin/env julia
"""
Scratch script to test BRDF plotting functionality

This script loads a BRDF file and creates a hemisphere plot
to investigate any issues with the visualization.
"""

println("=" ^ 70)
println("BRDF Plotting Test Script")
println("=" ^ 70)

# Activate the project environment
using Pkg
Pkg.activate(".")

println("\n📦 Loading BRDFPlatform...")
using BRDFPlatform
using WGLMakie

# Test file path - using a simple material
test_file = raw"C:\Users\makle\Desktop\Uni\BA\BRDFToolkit\test\MERL Database\chrome.binary"

println("\n📁 Loading BRDF file: $test_file")
println("-" ^ 70)

# Load the BRDF data
@info "Calling load_brdf..."
data = load_brdf(test_file)

# Check if loaded successfully
if data.status != :ok
    println("❌ Failed to load BRDF data!")
    println("   Error: $(data.error_msg)")
    exit(1)
end

println("✅ BRDF loaded successfully!")
println("   Name: $(data.name)")
println("   Format: $(data.source_format)")
println("   Wavelengths: $(data.wavelengths)")

# Print sampling info from internal data
println("   Samples: $(BRDFPlatform.n_samples_str(data._data))")

println("\n" ^ 70)
println("🎨 Creating visualizations...")
println("-" ^ 70)

# Set incident direction (45 degrees from normal, phi=0)
wi_theta = Float32(π/4)  # 45 degrees
wi_phi = Float32(0.0)

@info "Incident angles" theta_deg=rad2deg(wi_theta) phi_deg=rad2deg(wi_phi)

# ============================================================================
# Test 1: Hemisphere Plot
# ============================================================================
println("\n📊 Test 1: Generating hemisphere plot...")
try
    fig_hemisphere = create_hemisphere_plot(data, wi_theta, wi_phi, resolution=30)

    println("✅ Hemisphere plot created successfully!")

    # Save the plot
    output_file = "test_hemisphere_plot.png"
    save(output_file, fig_hemisphere)
    println("💾 Saved to: $output_file")

    # Display
    display(fig_hemisphere)

catch e
    println("❌ Error creating hemisphere plot!")
    showerror(stdout, e, catch_backtrace())
    println()
end

# ============================================================================
# Test 2: Polar Slice Plot
# ============================================================================
println("\n📊 Test 2: Generating polar slice plot...")
println("   This shows BRDF(φ_out) for multiple θ_out values")

try
    # Define outgoing polar angles
    theta_out_degrees = [15, 30, 45, 60, 75]
    theta_out_values = Float32.(deg2rad.(theta_out_degrees))

    @info "Creating polar slices" theta_degrees=theta_out_degrees

    fig_polar = create_polar_slice_plot(data, wi_theta, wi_phi,
                                        theta_out_values=theta_out_values,
                                        n_phi_samples=180)

    println("✅ Polar slice plot created successfully!")

    # Save the plot
    output_file = "test_polar_slice_plot.png"
    save(output_file, fig_polar)
    println("💾 Saved to: $output_file")

    # Display
    display(fig_polar)

catch e
    println("❌ Error creating polar slice plot!")
    showerror(stdout, e, catch_backtrace())
    println()
end

println("\n✨ Script completed!")
println("=" ^ 70)

# Keep the script alive to view the plots
println("\n⏸️  Press Enter to exit...")
readline()
