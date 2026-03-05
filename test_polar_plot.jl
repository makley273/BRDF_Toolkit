#!/usr/bin/env julia
"""
Test script for polar slice BRDF visualization

This script creates polar plots showing BRDF values as a function of
outgoing azimuthal angle for different outgoing polar angles.
"""

println("=" ^ 70)
println("BRDF Polar Slice Plot Test")
println("=" ^ 70)

# Activate the project environment
using Pkg
Pkg.activate(".")

println("\n📦 Loading BRDFPlatform...")
using BRDFPlatform
using WGLMakie

# Test files - try different materials
test_files = [
    (raw"/run/media/maka/data/BRDFDatabase/MERL/green-metallic-paint.binary", "green-metallic-paint"),
    (raw"/run/media/maka/data/BRDFDatabase/MERL/color-changing-paint2.binary", "color-changing-paint2"),
    (raw"C:\Users\makle\Desktop\Uni\BA\BRDFToolkit\test\MERL Database\teflon.binary", "teflon"),
]

# Choose which file to test
test_file, material_name = test_files[1]  # Chrome

println("\n📁 Loading BRDF file: $material_name")
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

println("\n" ^ 70)
println("🎨 Creating polar slice plot...")
println("-" ^ 70)

# Set incident direction (45 degrees from normal, phi=0)
wi_theta = Float32(π/4)  # 45 degrees
wi_phi = Float32(0.0)

# Define outgoing polar angles to plot (in degrees, then convert to radians)
theta_out_degrees = [15, 30, 45, 60, 75]
theta_out_values = Float32.(deg2rad.(theta_out_degrees))

@info "Incident angles" theta_deg=rad2deg(wi_theta) phi_deg=rad2deg(wi_phi)
@info "Outgoing theta slices" theta_degrees=theta_out_degrees

# Create the polar plot
println("\n📊 Generating polar visualization...")
println("   This will create 4 polar plots (R, G, B, Luminance)")
println("   Each shows BRDF(φ_out) for θ_out ∈ {$(join(theta_out_degrees, "°, "))°}")

# try
using Infiltrator
# @infiltrate 
fig = create_polar_slice_plot(data, wi_theta, wi_phi,
                                   theta_out_values=theta_out_values,
                                   n_phi_samples=720)

#     println("\n✅ Polar plot created successfully!")
#     println("\n💾 Saving plot to file...")

#     # Save the plot
#     # output_file = "test_polar_slice_$(material_name).png"
#     # save(output_file, fig)

#     # println("✅ Plot saved to: $output_file")

#     # Display the interactive plot
#     println("\n🖼️  Displaying interactive plot...")
#     println("   💡 Hover over data points to see values!")
display(fig)

#     println("\n✨ Script completed successfully!")
#     println("=" ^ 70)

# catch e
#     println("\n❌ Error creating polar plot!")
#     println("Exception: $e")
#     showerror(stdout, e, catch_backtrace())
#     println()
#     exit(1)
# end

# Keep the script alive to view the plot
println("\n⏸️  Press Enter to exit...")
readline()
