// Cook-Torrance microfacet BRDF with energy-conserving Fresnel-weighted diffuse
//
// f(wi, wo) = (1 - F(wi,h)) * albedo/PI  +  F(wi,h) * D(h) * G(wi,wo) / (4 * NdotL * NdotV)
//
// The (1 - F) factor ensures energy conservation: as the specular Fresnel term
// approaches 1 at grazing angles, the diffuse contribution fades to zero.
// Per-channel (RGB) Fresnel is used so albedo tinting is modulated per wavelength.
//
// D: Beckmann normal distribution function
// G: V-cavity geometric attenuation
// F: Schlick Fresnel approximation (per channel)
//
// Roughness parameterisation: α = roughness², α² = roughness⁴  (Disney convention)
//
// Uniforms:
//   u_roughness : float — perceptual roughness ∈ [0, 1]
//   u_F0        : vec3  — Fresnel reflectance at normal incidence
//   u_albedo    : vec3  — diffuse reflectance colour

uniform float u_roughness;
uniform vec3  u_F0;
uniform vec3  u_albedo;

const float PI = 3.14159265358979323846;

float beckmann_d(float NdotH, float alpha2) {
    float NdotH2 = max(NdotH * NdotH, 1e-6);
    float tan2   = (1.0 - NdotH2) / NdotH2;
    return exp(-tan2 / alpha2) / (PI * alpha2 * NdotH2 * NdotH2);
}

float vcavity_g(float NdotL, float NdotV, float NdotH, float VdotH) {
    float vh = max(VdotH, 1e-6);
    return min(1.0, min(2.0 * NdotH * NdotV / vh, 2.0 * NdotH * NdotL / vh));
}

vec3 schlick(vec3 F0, float VdotH) {
    float x  = 1.0 - VdotH;
    float x5 = x * x * x * x * x;
    return F0 + (1.0 - F0) * x5;
}

vec3 brdf_evaluate(vec3 wi, vec3 wo, vec3 N) {
    float NdotL = dot(wi, N);
    float NdotV = dot(wo, N);
    if (NdotL <= 0.0 || NdotV <= 0.0) return vec3(0.0);

    vec3  h     = normalize(wi + wo);
    float NdotH = clamp(dot(h, N),  0.0, 1.0);
    float VdotH = clamp(dot(wo, h), 0.0, 1.0);

    // Disney roughness remap: α = roughness², α² = roughness⁴
    float alpha  = u_roughness * u_roughness;
    float alpha2 = max(alpha * alpha, 1e-6);

    float D = beckmann_d(NdotH, alpha2);
    float G = vcavity_g(NdotL, NdotV, NdotH, VdotH);
    vec3  F = schlick(u_F0, VdotH);

    vec3 spec = D * G * F / (4.0 * NdotL * NdotV);
    vec3 diff = (1.0 - F) * u_albedo / PI;

    return max(vec3(0.0), diff + spec);
}
