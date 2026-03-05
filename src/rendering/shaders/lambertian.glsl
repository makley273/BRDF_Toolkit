// Lambertian (ideal diffuse) BRDF
//
// f(wi, wo) = albedo / PI
//
// Uniforms:
//   u_albedo: vec3 — diffuse reflectance per channel

uniform vec3 u_albedo;

vec3 brdf_evaluate(vec3 wi, vec3 wo, vec3 N) {
    if (dot(wi, N) <= 0.0 || dot(wo, N) <= 0.0) return vec3(0.0);
    return u_albedo / 3.14159265358979323846;
}
