// Anisotropic GGX microfacet BRDF
//
// f(wi, wo) = albedo/PI  +  D(h) * G(wi,wo) * F(wi,h) / (4 * NdotL * NdotV)
//
// D: anisotropic GGX NDF (separate αx, αy along tangent / bitangent)
// G: anisotropic Smith-GGX masking-shadowing
// F: Schlick Fresnel approximation
//
// A tangent frame (T, B) is derived from the surface normal here. For
// production use, supply a stable per-vertex tangent attribute instead.
//
// Uniforms:
//   u_roughness_x : float
//   u_roughness_y : float
//   u_F0          : vec3
//   u_albedo      : vec3

uniform float u_roughness_x;
uniform float u_roughness_y;
uniform vec3  u_F0;
uniform vec3  u_albedo;

const float PI = 3.14159265358979323846;

float ggx_d_aniso(float hx, float hy, float hz, float ax, float ay) {
    float ax_s = max(ax, 1e-4);
    float ay_s = max(ay, 1e-4);
    float r = (hx / ax_s) * (hx / ax_s) + (hy / ay_s) * (hy / ay_s) + hz * hz;
    return 1.0 / (PI * ax_s * ay_s * r * r);
}

float smith_g1_aniso(float vx, float vy, float vz, float ax, float ay) {
    float vz_s = max(vz, 1e-6);
    float lam_arg = (ax * ax * vx * vx + ay * ay * vy * vy) / (vz_s * vz_s);
    float lam = 0.5 * (-1.0 + sqrt(1.0 + lam_arg));
    return 1.0 / (1.0 + lam);
}

vec3 schlick(vec3 F0, float VdotH) {
    float x  = 1.0 - VdotH;
    float x5 = x * x * x * x * x;
    return F0 + (1.0 - F0) * x5;
}

// Orthonormal tangent frame aligned with world axes when N = +Z, matching
// the Julia shading-frame convention (tangent = +X, bitangent = +Y).
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
    float VdotH = clamp(dot(wo, h), 0.0, 1.0);

    vec3 T, B;
    build_tangent_frame(N, T, B);
    // Project halfvector and directions into the tangent frame.
    float hx = dot(h,  T); float hy = dot(h,  B); float hz = dot(h,  N);
    float ix = dot(wi, T); float iy = dot(wi, B); float iz = NdotL;
    float ox = dot(wo, T); float oy = dot(wo, B); float oz = NdotV;

    float ax = max(u_roughness_x * u_roughness_x, 1e-4);
    float ay = max(u_roughness_y * u_roughness_y, 1e-4);

    float D = ggx_d_aniso(hx, hy, hz, ax, ay);
    float G = smith_g1_aniso(ix, iy, iz, ax, ay) *
              smith_g1_aniso(ox, oy, oz, ax, ay);
    vec3  F = schlick(u_F0, VdotH);

    vec3 spec = D * G * F / (4.0 * NdotL * NdotV);
    vec3 diff = u_albedo / PI;

    return max(vec3(0.0), diff + spec);
}
