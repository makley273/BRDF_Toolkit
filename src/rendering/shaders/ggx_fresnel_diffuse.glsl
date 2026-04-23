// GGX microfacet BRDF with energy-conserving Fresnel-weighted diffuse
//
// f(wi, wo) = (1 - F(wi,h)) * albedo/PI  +  D(h) * G(wi,wo) * F(wi,h) / (4 * NdotL * NdotV)
//
// The (1 - F) factor ensures energy conservation: as the specular Fresnel term
// approaches 1 at grazing angles, the diffuse contribution fades to zero.
// Per-channel (RGB) Fresnel is used so albedo tinting is modulated per wavelength.
//
// D: GGX normal distribution function
// G: Smith-GGX geometry term (separable)
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

float ggx_d(float NdotH, float alpha2) {
    float d = NdotH * NdotH * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * d * d);
}

float smith_g1(float NdotV, float alpha2) {
    return 2.0 * NdotV / (NdotV + sqrt(alpha2 + (1.0 - alpha2) * NdotV * NdotV));
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

    float alpha  = u_roughness * u_roughness;
    float alpha2 = max(alpha * alpha, 1e-6);

    float D = ggx_d(NdotH, alpha2);
    float G = smith_g1(NdotL, alpha2) * smith_g1(NdotV, alpha2);
    vec3  F = schlick(u_F0, VdotH);

    vec3 spec = D * G * F / (4.0 * NdotL * NdotV);
    vec3 diff = (1.0 - F) * u_albedo / PI;

    return max(vec3(0.0), diff + spec);
}
