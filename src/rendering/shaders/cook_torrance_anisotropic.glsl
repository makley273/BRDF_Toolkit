// Anisotropic Cook-Torrance microfacet BRDF
//
// f(wi, wo) = albedo/PI  +  F(wi,h) * D(h) * G(wi,wo) / (4 * NdotL * NdotV)
//
// D: anisotropic Beckmann NDF (separate αx, αy along tangent / bitangent)
// G: V-cavity geometric attenuation
// F: Schlick Fresnel approximation
//
// A tangent frame (T, B) is derived from the surface normal here. For
// production use, supply a stable per-vertex tangent attribute instead.
//
// Uniforms:
//   u_roughness_x : float — tangent-direction perceptual roughness ∈ [0, 1]
//   u_roughness_y : float — bitangent-direction perceptual roughness ∈ [0, 1]
//   u_F0          : vec3
//   u_albedo      : vec3

uniform float u_roughness_x;
uniform float u_roughness_y;
uniform vec3  u_F0;
uniform vec3  u_albedo;

const float PI = 3.14159265358979323846;

float beckmann_d_aniso(float hx, float hy, float hz, float ax, float ay) {
    float ax_s = max(ax, 1e-4);
    float ay_s = max(ay, 1e-4);
    float hz2  = max(hz * hz, 1e-6);
    float e    = -((hx * hx) / (ax_s * ax_s) + (hy * hy) / (ay_s * ay_s)) / hz2;
    return exp(e) / (PI * ax_s * ay_s * hz2 * hz2);
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

// Derive an orthonormal tangent frame from N. Chosen so that when N = +Z
// the frame is (T = +X, B = +Y, N = +Z) — i.e. it matches the shading-frame
// convention used by the Julia evaluate() implementations.
void build_tangent_frame(vec3 N, out vec3 T, out vec3 B) {
    vec3 up = abs(N.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    T = normalize(cross(up, N));
    B = cross(N, T);
}

vec3 brdf_evaluate(vec3 wi, vec3 wo, vec3 N) {
    float NdotL = dot(wi, N);
    float NdotV = dot(wo, N);
    if (NdotL <= 0.0 || NdotV <= 0.0) return vec3(0.0);

    vec3  h     = normalize(wi + wo);
    float NdotH = clamp(dot(h, N),  0.0, 1.0);
    float VdotH = clamp(dot(wo, h), 0.0, 1.0);

    vec3 T, B;
    build_tangent_frame(N, T, B);
    float hx = dot(h, T);
    float hy = dot(h, B);
    float hz = NdotH;

    float ax = max(u_roughness_x * u_roughness_x, 1e-4);
    float ay = max(u_roughness_y * u_roughness_y, 1e-4);

    float D = beckmann_d_aniso(hx, hy, hz, ax, ay);
    float G = vcavity_g(NdotL, NdotV, NdotH, VdotH);
    vec3  F = schlick(u_F0, VdotH);

    vec3 spec = D * G * F / (4.0 * NdotL * NdotV);
    vec3 diff = u_albedo / PI;

    return max(vec3(0.0), diff + spec);
}
