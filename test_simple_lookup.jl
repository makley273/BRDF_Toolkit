#!/usr/bin/env julia
"""
Simple test script to verify BRDF loading and lookup work correctly
"""

println("=" ^ 70)
println("Simple BRDF Lookup Test")
println("=" ^ 70)

using Pkg
Pkg.activate(".")

println("\n📦 Loading BRDFPlatform...")
using BRDFPlatform

# Test file
test_file = raw"C:\Users\makle\Desktop\Uni\BA\BRDFToolkit\test\MERL Database\chrome.binary"

println("\n📁 Loading BRDF: $test_file")
data = load_brdf(test_file)

if data.status != :ok
    println("❌ Failed: $(data.error_msg)")
    exit(1)
end

println("✅ Loaded: $(data.name)")
println("   Format: $(data.source_format)")
println("   Wavelengths: $(data.wavelengths)")

# Test a few lookups
println("\n🔍 Testing BRDF lookups...")

# Normal incidence, normal viewing
wi = (0.0f0, 0.0f0, 1.0f0)
wo = (0.0f0, 0.0f0, 1.0f0)
result = brdf_value(data, wi, wo)
println("   Normal → Normal: RGB$(result)")

# 45 degree incidence, specular reflection
wi = (sin(π/4), 0.0f0, cos(π/4))
wo = (sin(π/4), 0.0f0, cos(π/4))
result = brdf_value(data, wi, wo)
println("   45° → 45° (specular): RGB$(result)")

# 45 degree incidence, normal viewing
wi = (sin(π/4), 0.0f0, cos(π/4))
wo = (0.0f0, 0.0f0, 1.0f0)
result = brdf_value(data, wi, wo)
println("   45° → Normal: RGB$(result)")

# Grazing incidence
wi = (sin(π/3), 0.0f0, cos(π/3))
wo = (sin(π/3), 0.0f0, cos(π/3))
result = brdf_value(data, wi, wo)
println("   60° → 60° (specular): RGB$(result)")

println("\n✅ Lookup test complete!")
println("=" ^ 70)
