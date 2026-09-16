// Universal measured/fitted BRDF evaluation via a 4D Rusinkiewicz lookup table.
//
// The 4D grid (theta_h, theta_d, phi_d, phi_h) is stored in a 3D texture by
// folding the anisotropy axis (phi_h) and phi_d together along depth:
//
//     texture dims = (N_th, N_td, N_pd * N_ph)
//     depth index  z = phi_h_block * N_pd + phi_d
//
// Hardware trilinear filtering interpolates (theta_h, theta_d, phi_d); we add a
// manual lerp across phi_h (the folded fourth axis, periodic). For isotropic data
// every phi_h block is identical and the manual lerp is a no-op.
//
// Forward angle warps here are the exact inverse of the LUT generator in
// src/rendering/universal_texture.jl.

const float U_PI = 3.14159265359;

uniform vec4  uGridDims;   // (N_theta_h, N_theta_d, N_phi_d, N_phi_h)
uniform float uIntensity;  // cosine-weighted display gain (π ≈ true-albedo reflectance)

// theta_h uses the sqrt warp (concentrates samples near the specular peak).
float u_idx_theta_h(float th, float n) {
    float v = sqrt(clamp(th / (U_PI * 0.5), 0.0, 1.0)) * (n - 1.0);
    return clamp(v, 0.0, n - 1.0);
}
float u_idx_linear(float a, float maxAngle, float n) {
    return clamp(a / maxAngle * (n - 1.0), 0.0, n - 1.0);
}

// Hardware-trilinear sample of one phi_h block `a` (integer) at fractional
// (theta_h, theta_d, phi_d) coordinates. Clamping fp to [0, N_pd-1] keeps the
// GPU's depth interpolation from bleeding into the neighbouring block.
vec3 u_sample_block(float fth, float ftd, float fp, float a) {
    float N_th = uGridDims.x, N_td = uGridDims.y, N_pd = uGridDims.z, N_ph = uGridDims.w;
    float x = (fth + 0.5) / N_th;
    float y = (ftd + 0.5) / N_td;
    float z = (a * N_pd + fp + 0.5) / (N_pd * N_ph);
    return texture(brdfData, vec3(x, y, z)).rgb;
}

vec3 brdf_evaluate(vec3 wi, vec3 wo, vec3 N) {
    float NdotL = dot(wi, N);
    float NdotV = dot(wo, N);
    if (NdotL <= 0.0 || NdotV <= 0.0) return vec3(0.0);

    vec3 H = normalize(wi + wo);

    float theta_h = acos(clamp(dot(N, H), -1.0, 1.0));
    float theta_d = acos(clamp(dot(wi, H), -1.0, 1.0));

    // phi_d: azimuth of wi in a frame whose pole is H (mirrors the LUT generator).
    vec3 t = N - dot(N, H) * H;
    float tl = length(t);
    if (tl < 1e-6) {
        t = vec3(1.0, 0.0, 0.0) - H.x * H;   // H ∥ N: arbitrary tangent ⊥ H
        tl = length(t);
    }
    t /= tl;
    vec3 b = cross(H, t);
    
    // Full azimuth [0, 2π) — NO symmetry fold. Halving this range needs either
    // reciprocity (f(phi_d) == f(phi_d + π)) or bilateral symmetry
    // (f(phi_d) == f(-phi_d)); the latter holds only for ISOTROPIC materials, so
    // folding with abs() mirrors half the azimuth onto the wrong data for
    // anisotropic measurements (UTIA, RGL) and paints radial spokes around the
    // highlight. Measured tables satisfy neither symmetry exactly, so the LUT
    // stores the whole circle and the shader just wraps into it.
    float phi_d = atan(dot(wi, b), dot(wi, t));
    if (phi_d < 0.0) phi_d += 2.0 * U_PI;

    // phi_h: azimuth of H in a surface tangent frame. The grain runs along the
    // longitudes of a world-up (0,1,0) sphere, so anisotropy flows pole-to-pole
    // and the frame is smooth everywhere except the two Y-poles (top/bottom of
    // the silhouette). NO branch on N — a branch produces a visible discontinuity
    // ring. Must match _angles_to_dirs in universal_texture.jl exactly.
    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 T = cross(up, N);
    float tLen = length(T);
    T = (tLen < 1e-5) ? vec3(1.0, 0.0, 0.0) : T / tLen;   // fallback only at the poles
    vec3 Bt = cross(N, T);
    
    float phi_h = atan(dot(H, Bt), dot(H, T));
    if (phi_h < 0.0) phi_h += 2.0 * U_PI;    // [0, 2π)

    // Angles → continuous grid coordinates.
    float N_th = uGridDims.x, N_td = uGridDims.y, N_pd = uGridDims.z, N_ph = uGridDims.w;
    float fth = u_idx_theta_h(theta_h, N_th);
    float ftd = u_idx_linear(theta_d, U_PI * 0.5, N_td);
    float fp  = u_idx_linear(phi_d,  2.0 * U_PI,  N_pd);

    // phi_h is periodic: blend two adjacent blocks (hardware trilinear each).
    // The manual lerp perfectly handles the 2π -> 0 wrap without texture jumping.
    float fa = phi_h / (2.0 * U_PI) * N_ph;   // [0, N_ph)
    float a0 = floor(fa);
    float frac = fa - a0;
    float a0w = mod(a0, N_ph);
    float a1w = mod(a0 + 1.0, N_ph);

    vec3 s0 = u_sample_block(fth, ftd, fp, a0w);
    vec3 s1 = u_sample_block(fth, ftd, fp, a1w);
    vec3 brdfValue = mix(s0, s1, frac);

    // Unlit mode (uUnlit=1, declared by renderer.js): the LUT holds display
    // colors (error maps) — return them unshaded so the colormap reads true.
    return brdfValue * mix(NdotL * uIntensity, 1.0, uUnlit);
}