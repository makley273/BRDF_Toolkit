"""
Test MERL loader against reference merl.exe output

This script tests if our Julia MERL loader produces the same values
as the reference merl.exe tool.

Reference test case (from merl.exe - example.txt):
  File: blue-acrylic.binary
  Angles: theta_in=45°, phi_in=45°, theta_out=60°, phi_out=70°
  Expected: Red=0.003357, Green=0.011261, Blue=0.033194
  Indices: theta_h=51.84°, phi_h=58.78°, theta_d=12.35°, phi_d=-128.02°
"""

using Test
using LinearAlgebra

# Add parent directory to path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using BRDFPlatform

# Reference values from merl.exe output
const TEST_FILE = joinpath(@__DIR__, "MERL Database", "blue-acrylic.binary")
const THETA_IN = 45.0  # degrees
const PHI_IN = 45.0    # degrees
const THETA_OUT = 60.0 # degrees
const PHI_OUT = 70.0   # degrees

const EXPECTED_RED = 0.003357
const EXPECTED_GREEN = 0.011261
const EXPECTED_BLUE = 0.033194

# Convert to radians
const θ_i = deg2rad(THETA_IN)
const φ_i = deg2rad(PHI_IN)
const θ_o = deg2rad(THETA_OUT)
const φ_o = deg2rad(PHI_OUT)

# Convert spherical angles to direction vectors
function spherical_to_cartesian(theta::Float64, phi::Float64)
    x = sin(theta) * cos(phi)
    y = sin(theta) * sin(phi)
    z = cos(theta)
    return [x, y, z]
end

# Convert to Rusinkiewicz half-angle parameterization
function to_rusinkiewicz(wi, wo)
    # Half-angle vector
    h = normalize(wi + wo)

    # Difference vector (in coordinate frame aligned with h)
    # This requires rotating wi and wo into a frame where h is the z-axis
    theta_h = acos(clamp(h[3], -1.0, 1.0))
    phi_h = atan(h[2], h[1])

    # For theta_diff and phi_diff, we need to compute the angle between wi and h
    theta_d = acos(clamp(dot(wi, h), -1.0, 1.0))

    # Compute phi_diff (rotation around h)
    # This is complex - simplified version
    phi_d = 0.0  # Placeholder

    return (theta_h, phi_h, theta_d, phi_d)
end

println("=" ^ 70)
println("Testing MERL Loader")
println("=" ^ 70)

# Load the BRDF
println("\n1. Loading BRDF from: $TEST_FILE")
brdf = load_brdf(BRDFPlatform.MERLFormat, TEST_FILE)

println("   Status: $(brdf.status)")
println("   Name: $(brdf.name)")
println("   Wavelengths: $(brdf.wavelengths)")
println("   Data type: $(typeof(brdf._data))")
println("   Samples: $(BRDFPlatform.n_samples_str(brdf._data))")

if brdf.status != :ok
    println("   ERROR: $(brdf.error_msg)")
    exit(1)
end

# Create direction vectors
println("\n2. Computing direction vectors")
wi = spherical_to_cartesian(θ_i, φ_i)
wo = spherical_to_cartesian(θ_o, φ_o)

println("   wi (θ=$(THETA_IN)deg, φ=$(PHI_IN)deg) = $wi")
println("   wo (θ=$(THETA_OUT)deg, φ=$(PHI_OUT)deg) = $wo")

# Compute Rusinkiewicz coordinates
println("\n3. Computing Rusinkiewicz coordinates")
theta_h, phi_h, theta_d, phi_d = to_rusinkiewicz(wi, wo)

println("   theta_h = $(rad2deg(theta_h))°")
println("   phi_h = $(rad2deg(phi_h))°")
println("   theta_d = $(rad2deg(theta_d))°")
println("   phi_d = $(rad2deg(phi_d))°")

println("\n   Expected (from merl.exe):")
println("   theta_h = 51.84°")
println("   phi_h = 58.78°")
println("   theta_d = 12.35°")
println("   phi_d = -128.02°")

# For now, let's directly test the index calculation
# using the known Rusinkiewicz coordinates from merl.exe
println("\n4. Testing index calculation")

# Use merl.exe reference angles in radians
ref_theta_h = deg2rad(51.84)
ref_theta_d = deg2rad(12.35)
ref_phi_d = deg2rad(-128.02)

# Note: Index calculation functions are now internal to the format implementation
# We test using the public lookup interface instead
println("   (Index calculation is now handled internally by lookup function)")

# Access the BRDF values
println("\n5. Reading BRDF values from loaded data")

# The samples array is [wavelength × sample_idx × 1]
# We need to convert (theta_h, theta_d, phi_d) to linear index
# Use lookup interface with the test directions
wi_test = BRDFPlatform.vec3(Float32(wi[1]), Float32(wi[2]), Float32(wi[3]))
wo_test = BRDFPlatform.vec3(Float32(wo[1]), Float32(wo[2]), Float32(wo[3]))

rgb_val = BRDFPlatform.brdf_value(brdf, wi_test, wo_test)
red_val = rgb_val.r
green_val = rgb_val.g
blue_val = rgb_val.b

println("\n6. Comparing values using lookup interface")
println("   Test directions: θi=$(THETA_IN)°, φi=$(PHI_IN)°, θo=$(THETA_OUT)°, φo=$(PHI_OUT)°")
println("   Channel  | Julia      | merl.exe   | Difference")
println("   " * "-" ^ 55)
println("   Red      | $red_val  | $EXPECTED_RED | $(abs(red_val - EXPECTED_RED))")
println("   Green    | $green_val | $EXPECTED_GREEN | $(abs(green_val - EXPECTED_GREEN))")
println("   Blue     | $blue_val  | $EXPECTED_BLUE | $(abs(blue_val - EXPECTED_BLUE))")

# Test with tolerance
tolerance = 1e-5
@testset "MERL Loader Validation" begin
    @test isapprox(red_val, EXPECTED_RED, atol=tolerance)
    @test isapprox(green_val, EXPECTED_GREEN, atol=tolerance)
    @test isapprox(blue_val, EXPECTED_BLUE, atol=tolerance)
end

println("\n" * "=" ^ 70)
println("Test Complete!")
println("=" ^ 70)
