using BRDFPlatform

brdf = load_brdf(BRDFPlatform.MERLFormat, "test/MERL Database/blue-acrylic.binary")

println("Testing MERL loader using lookup interface")
println("="^60)

# Test case 1: Normal incidence (wi = wo = normal)
println("\nTest 1: Normal incidence")
println("Expected (from merl.exe): R=16.707445, G=12.342452, B=6.751064")
wi_normal = BRDFPlatform.vec3(0.0f0, 0.0f0, 1.0f0)
wo_normal = BRDFPlatform.vec3(0.0f0, 0.0f0, 1.0f0)
rgb = BRDFPlatform.brdf_value(brdf, wi_normal, wo_normal)
println("Julia (lookup):            R=$(rgb.r), G=$(rgb.g), B=$(rgb.b)")

# Test case 2: The problematic case - complex angles
# θi=45°, φi=45°, θo=60°, φo=70°
println("\nTest 2: Complex angle (θi=45°, φi=45°, θo=60°, φo=70°)")
println("Expected: R=0.003357, G=0.011261, B=0.033194")

# Convert angles to directions (away from surface)
theta_i = deg2rad(45.0f0)
phi_i = deg2rad(45.0f0)
theta_o = deg2rad(60.0f0)
phi_o = deg2rad(70.0f0)

wi = BRDFPlatform.vec3(
    sin(theta_i) * cos(phi_i),
    sin(theta_i) * sin(phi_i),
    cos(theta_i)
)

wo = BRDFPlatform.vec3(
    sin(theta_o) * cos(phi_o),
    sin(theta_o) * sin(phi_o),
    cos(theta_o)
)

rgb2 = BRDFPlatform.brdf_value(brdf, wi, wo)
println("Julia (lookup):       R=$(rgb2.r), G=$(rgb2.g), B=$(rgb2.b)")
