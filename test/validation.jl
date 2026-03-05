"""
Julia vs GLSL Validation Tests

Strategy: re-implement each GLSL shader's math in Julia (using Float32
throughout, matching GLSL's default precision) and compare against the
model's evaluate() function.

This validates that the two code-paths are mathematically equivalent.
It does NOT test GLSL compilation — that requires a headless browser
(future work with Playwright.jl).

Error thresholds (from CLAUDE.md):
  - Lambertian : max relative error < 1e-4
  - GGX        : max relative error < 1e-3
  - Cook-Torrance : max relative error < 1e-3
"""

using Test
using Random
using StaticArrays
using Colors
using LinearAlgebra

# Load the platform (adjust path if running standalone)
project_root = joinpath(@__DIR__, "..")
if !isdefined(Main, :BRDFPlatform)
    import Pkg
    Pkg.activate(project_root)
    using BRDFPlatform
end

# ============================================================================
# Helpers
# ============================================================================

"""Generate N random unit directions in the upper hemisphere."""
function random_upper_hemisphere(n::Int; seed::Int=42)
    rng = MersenneTwister(seed)
    dirs = Vector{SVector{3, Float32}}(undef, n)
    i = 1
    while i <= n
        v = SVector{3, Float32}(randn(rng, Float32), randn(rng, Float32), abs(randn(rng, Float32)))
        len = norm(v)
        len < 1f-6 && continue
        dirs[i] = v / len
        i += 1
    end
    return dirs
end

"""
Relative error between two RGB values.
Uses (|a - b| / (|b| + ε)) per channel, returns the max across channels.
"""
function max_relative_error(a::RGB{Float32}, b::RGB{Float32}; eps::Float32=1f-6)::Float32
    function rel(x, y)
        abs(x - y) / (abs(y) + eps)
    end
    return max(rel(a.r, b.r), rel(a.g, b.g), rel(a.b, b.b))
end

# ============================================================================
# GLSL reference implementations in Julia/Float32
#
# These mirror the GLSL shaders exactly. Any divergence from evaluate()
# reveals a bug in either the Julia or GLSL code.
# ============================================================================

# --- Shared microfacet helpers (mirror of GLSL helper functions) ---

@inline function glsl_ggx_d(NdotH::Float32, alpha2::Float32)::Float32
    d = NdotH * NdotH * (alpha2 - 1.0f0) + 1.0f0
    return alpha2 / (Float32(π) * d * d)
end

@inline function glsl_smith_g1(NdotV::Float32, alpha2::Float32)::Float32
    return 2.0f0 * NdotV / (NdotV + sqrt(alpha2 + (1.0f0 - alpha2) * NdotV * NdotV))
end

@inline function glsl_schlick(F0::SVector{3, Float32}, VdotH::Float32)::SVector{3, Float32}
    x  = 1.0f0 - VdotH
    x5 = x * x * x * x * x
    return F0 .+ (1.0f0 .- F0) .* x5   # broadcast: scalar ± SVector
end

# --- Lambertian GLSL reference ---

"""
Mirror of lambertian.glsl::brdf_evaluate with N = (0,0,1).
"""
function glsl_lambertian(u_albedo::SVector{3, Float32},
                          wi::SVector{3, Float32},
                          wo::SVector{3, Float32})::RGB{Float32}
    N = SVector{3, Float32}(0, 0, 1)
    if dot(wi, N) <= 0.0f0 || dot(wo, N) <= 0.0f0
        return RGB{Float32}(0, 0, 0)
    end
    v = u_albedo / Float32(π)
    return RGB{Float32}(v[1], v[2], v[3])
end

# --- GGX GLSL reference ---

"""
Mirror of ggx.glsl::brdf_evaluate with N = (0,0,1).
"""
function glsl_ggx(u_roughness::Float32,
                   u_F0::SVector{3, Float32},
                   wi::SVector{3, Float32},
                   wo::SVector{3, Float32})::RGB{Float32}
    N = SVector{3, Float32}(0, 0, 1)
    NdotL = dot(wi, N)
    NdotV = dot(wo, N)
    if NdotL <= 0.0f0 || NdotV <= 0.0f0
        return RGB{Float32}(0, 0, 0)
    end

    h     = normalize(wi + wo)
    NdotH = clamp(dot(h, N),  0.0f0, 1.0f0)
    VdotH = clamp(dot(wo, h), 0.0f0, 1.0f0)

    alpha  = u_roughness * u_roughness
    alpha2 = max(alpha * alpha, 1.0f-6)

    D  = glsl_ggx_d(NdotH, alpha2)
    G  = glsl_smith_g1(NdotL, alpha2) * glsl_smith_g1(NdotV, alpha2)
    F  = glsl_schlick(u_F0, VdotH)

    spec = D * G * F / (4.0f0 * NdotL * NdotV)
    return RGB{Float32}(max(0.0f0, spec[1]), max(0.0f0, spec[2]), max(0.0f0, spec[3]))
end

# --- Cook-Torrance GLSL reference ---

"""
Mirror of cook_torrance.glsl::brdf_evaluate with N = (0,0,1).
"""
function glsl_cook_torrance(u_roughness::Float32,
                              u_F0::SVector{3, Float32},
                              u_albedo::SVector{3, Float32},
                              wi::SVector{3, Float32},
                              wo::SVector{3, Float32})::RGB{Float32}
    N = SVector{3, Float32}(0, 0, 1)
    NdotL = dot(wi, N)
    NdotV = dot(wo, N)
    if NdotL <= 0.0f0 || NdotV <= 0.0f0
        return RGB{Float32}(0, 0, 0)
    end

    h     = normalize(wi + wo)
    NdotH = clamp(dot(h, N),  0.0f0, 1.0f0)
    VdotH = clamp(dot(wo, h), 0.0f0, 1.0f0)

    alpha  = u_roughness * u_roughness
    alpha2 = max(alpha * alpha, 1.0f-6)

    D    = glsl_ggx_d(NdotH, alpha2)
    G    = glsl_smith_g1(NdotL, alpha2) * glsl_smith_g1(NdotV, alpha2)
    F    = glsl_schlick(u_F0, VdotH)
    spec = D * G * F / (4.0f0 * NdotL * NdotV)
    diff = u_albedo / Float32(π)
    out  = diff + spec
    return RGB{Float32}(max(0.0f0, out[1]), max(0.0f0, out[2]), max(0.0f0, out[3]))
end

# ============================================================================
# Test runner
# ============================================================================

"""
Compare model.evaluate() against a GLSL reference function over N random
direction pairs. Returns (max_error, mean_error, n_tested).
"""
function run_comparison(model_eval_fn, glsl_ref_fn, n_samples::Int=2000)
    wis = random_upper_hemisphere(n_samples; seed=1)
    wos = random_upper_hemisphere(n_samples; seed=2)

    errors = Float32[]
    for (wi, wo) in zip(wis, wos)
        julia_val = model_eval_fn(wi, wo)
        glsl_val  = glsl_ref_fn(wi, wo)

        # Skip near-zero outputs (relative error is unreliable there)
        max_val = max(julia_val.r, julia_val.g, julia_val.b,
                      glsl_val.r,  glsl_val.g,  glsl_val.b)
        max_val < 1f-8 && continue

        push!(errors, max_relative_error(julia_val, glsl_val))
    end

    isempty(errors) && return (0f0, 0f0, 0)
    return (maximum(errors), sum(errors) / length(errors), length(errors))
end

# ============================================================================
# Tests
# ============================================================================

@testset "Model Julia vs GLSL Validation" begin

    # ------------------------------------------------------------------
    @testset "Lambertian" begin
        albedo = RGB{Float32}(0.8, 0.5, 0.2)
        model  = LambertianModel(albedo)
        u_alb  = SVector{3, Float32}(albedo.r, albedo.g, albedo.b)

        max_err, mean_err, n = run_comparison(
            (wi, wo) -> evaluate(model, wi, wo),
            (wi, wo) -> glsl_lambertian(u_alb, wi, wo)
        )

        @info "Lambertian" max_err mean_err n_samples=n
        @test max_err < 1f-4

        # Analytical check: value must equal albedo/π everywhere in hemisphere
        wi = SVector{3, Float32}(0, 0, 1)
        wo = SVector{3, Float32}(sin(0.3f0), 0, cos(0.3f0))
        v  = evaluate(model, wi, wo)
        @test v.r ≈ Float32(albedo.r / π) atol=1f-6
        @test v.g ≈ Float32(albedo.g / π) atol=1f-6
        @test v.b ≈ Float32(albedo.b / π) atol=1f-6

        # Below-horizon: must return zero
        wi_below = SVector{3, Float32}(0, 0, -1)
        @test evaluate(model, wi_below, wo) == RGB{Float32}(0, 0, 0)
    end

    # ------------------------------------------------------------------
    @testset "GGX" begin
        for (roughness, F0_val) in [(0.1f0, 0.04f0), (0.5f0, 0.5f0), (0.9f0, 0.9f0)]
            F0    = RGB{Float32}(F0_val, F0_val, F0_val)
            model = GGXModel(roughness, F0)
            u_F0  = SVector{3, Float32}(F0_val, F0_val, F0_val)

            max_err, mean_err, n = run_comparison(
                (wi, wo) -> evaluate(model, wi, wo),
                (wi, wo) -> glsl_ggx(roughness, u_F0, wi, wo)
            )

            @info "GGX" roughness F0_val max_err mean_err n_samples=n
            @test max_err < 1f-3
        end

        # Below-horizon: must return zero
        model    = GGXModel(0.5f0, 0.04f0)
        wi_below = SVector{3, Float32}(0, 0, -1)
        wo_ok    = SVector{3, Float32}(0, 0,  1)
        @test evaluate(model, wi_below, wo_ok) == RGB{Float32}(0, 0, 0)
        @test evaluate(model, wo_ok, wi_below) == RGB{Float32}(0, 0, 0)
    end

    # ------------------------------------------------------------------
    @testset "Cook-Torrance" begin
        configs = [
            (roughness=0.1f0, F0=0.04f0, albedo=0.5f0),
            (roughness=0.5f0, F0=0.2f0,  albedo=0.2f0),
            (roughness=0.9f0, F0=0.9f0,  albedo=0.0f0),   # pure specular
        ]

        for cfg in configs
            F0    = RGB{Float32}(cfg.F0, cfg.F0, cfg.F0)
            alb   = RGB{Float32}(cfg.albedo, cfg.albedo, cfg.albedo)
            model = CookTorranceModel(cfg.roughness, F0, alb)
            u_F0  = SVector{3, Float32}(cfg.F0,    cfg.F0,    cfg.F0)
            u_alb = SVector{3, Float32}(cfg.albedo, cfg.albedo, cfg.albedo)

            max_err, mean_err, n = run_comparison(
                (wi, wo) -> evaluate(model, wi, wo),
                (wi, wo) -> glsl_cook_torrance(cfg.roughness, u_F0, u_alb, wi, wo)
            )

            @info "Cook-Torrance" roughness=cfg.roughness F0=cfg.F0 albedo=cfg.albedo max_err mean_err n_samples=n
            @test max_err < 1f-3
        end

        # Diffuse-only limit: F0=0, roughness→1 → should approach albedo/π
        # (not exact due to specular G/D terms, just check non-negative)
        model = CookTorranceModel(1.0f0, RGB{Float32}(0,0,0), RGB{Float32}(0.5f0,0.5f0,0.5f0))
        wi = SVector{3, Float32}(0, 0, 1)
        wo = SVector{3, Float32}(0, 0, 1)
        v  = evaluate(model, wi, wo)
        @test v.r >= 0f0 && v.g >= 0f0 && v.b >= 0f0

        # Below-horizon: must return zero
        wi_below = SVector{3, Float32}(0, 0, -1)
        @test evaluate(model, wi_below, wo) == RGB{Float32}(0, 0, 0)
    end

    # ------------------------------------------------------------------
    @testset "Energy conservation (informal)" begin
        # A physically plausible BRDF should not reflect more energy than
        # it receives. We check that the directional-hemispherical reflectance
        # is ≤ 1 for a sample of incident angles (Monte Carlo estimate).
        # This is not a strict guarantee but catches severe energy violations.

        n_mc = 4000
        wos  = random_upper_hemisphere(n_mc; seed=99)

        function estimate_dhr(model, wi::SVector{3, Float32})
            # ∫ f(wi,wo) cos(θo) dωo ≈ (2π/N) Σ f(wi,wo_i) * wo_i[3]
            # wo sampled uniformly on hemisphere → pdf = 1/(2π)
            total = RGB{Float32}(0, 0, 0)
            for wo in wos
                v = evaluate(model, wi, wo)
                total = RGB{Float32}(total.r + v.r * wo[3],
                                     total.g + v.g * wo[3],
                                     total.b + v.b * wo[3])
            end
            scale = Float32(2π / n_mc)
            return RGB{Float32}(total.r * scale, total.g * scale, total.b * scale)
        end

        wi_normal  = SVector{3, Float32}(0, 0, 1)
        wi_oblique = SVector{3, Float32}(sin(Float32(π/4)), 0, cos(Float32(π/4)))

        # Lambertian albedo=1 → DHR should be ≈ 1
        lamb = LambertianModel(1.0f0)
        dhr  = estimate_dhr(lamb, wi_normal)
        @test dhr.r < 1.05f0   # allow small Monte Carlo error

        # GGX and Cook-Torrance should not exceed 1
        for model in [GGXModel(0.3f0, 0.9f0),
                      CookTorranceModel(0.3f0, RGB{Float32}(0.04f0,0.04f0,0.04f0), RGB{Float32}(0.5f0,0.5f0,0.5f0))]
            for wi in [wi_normal, wi_oblique]
                dhr = estimate_dhr(model, wi)
                @test dhr.r <= 1.05f0
                @test dhr.g <= 1.05f0
                @test dhr.b <= 1.05f0
            end
        end
    end

end
