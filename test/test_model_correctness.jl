using Test
using BRDFPlatform
using StaticArrays, Colors, LinearAlgebra, Random

function sph(theta, phi)
    normalize(SVector{3,Float32}(
        sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)))
end

function mc_integral(model, wo; N=200_000, seed=42)
    rng = Xoshiro(seed)
    acc = RGB{Float64}(0,0,0)
    for _ in 1:N
        u1, u2 = rand(rng), rand(rng)
        theta  = acos(1.0 - u1)
        phi    = 2π * u2
        wi     = sph(theta, phi)
        f      = evaluate(model, wi, wo)
        NdotL  = wi[3]
        acc    = acc + RGB{Float64}(f.r, f.g, f.b) * NdotL * 2π
    end
    return acc / N
end

const wo_n  = SVector{3,Float32}(0f0, 0f0, 1f0)
const wo_30 = normalize(SVector{3,Float32}(0.5f0, 0f0, 0.866f0))

ALL_MODELS = [
    LambertianModel(RGB{Float32}(0.5f0,0.5f0,0.5f0)),
    CookTorranceModel(0.4f0, RGB{Float32}(0.04f0,0.04f0,0.04f0), RGB{Float32}(0.5f0,0.3f0,0.1f0)),
    CookTorranceSpecularModel(0.3f0, RGB{Float32}(0.9f0,0.7f0,0.1f0)),
    CookTorranceFresnelDiffuseModel(0.5f0, RGB{Float32}(0.04f0,0.04f0,0.04f0), RGB{Float32}(0.6f0,0.4f0,0.2f0)),
    CookTorranceScaledDiffuseModel(0.35f0, RGB{Float32}(0.05f0,0.05f0,0.05f0), RGB{Float32}(0.4f0,0.5f0,0.6f0), 0.6f0),
    GGXModel(0.4f0, RGB{Float32}(0.04f0,0.06f0,0.08f0), RGB{Float32}(0.5f0,0.3f0,0.1f0)),
    GGXSpecularModel(0.3f0, RGB{Float32}(0.9f0,0.7f0,0.1f0)),
    GGXFresnelDiffuseModel(0.5f0, RGB{Float32}(0.04f0,0.04f0,0.04f0), RGB{Float32}(0.6f0,0.4f0,0.2f0)),
    GGXScaledDiffuseModel(0.35f0, RGB{Float32}(0.05f0,0.05f0,0.05f0), RGB{Float32}(0.4f0,0.5f0,0.6f0), 0.6f0),
]

HEMISPHERE_DIRS = [sph(theta, phi)
    for theta in range(0.05f0, Float32(pi/2)-0.05f0, 8)
    for phi   in range(0.0f0, Float32(2pi), 12)]

RECIPROCITY_PAIRS = [
    (sph(0.3f0,0.5f0), sph(0.7f0,2.1f0)),
    (sph(1.0f0,1.0f0), sph(0.4f0,4.0f0)),
    (wo_n,             sph(0.6f0, 0.8f0)),
]

@testset "BRDF Model Correctness" begin

    @testset "Non-negativity" begin
        for m in ALL_MODELS
            name = nameof(typeof(m))
            fails = 0
            for wi in HEMISPHERE_DIRS, wo in HEMISPHERE_DIRS
                f = evaluate(m, wi, wo)
                if f.r < 0 || f.g < 0 || f.b < 0
                    fails += 1
                end
            end
            @test fails == 0
        end
    end

    @testset "Below-horizon returns zero" begin
        wi_below = SVector{3,Float32}(0f0, 0f0, -1f0)
        wo_below = SVector{3,Float32}(0f0, 0f0, -1f0)
        for m in ALL_MODELS
            @test evaluate(m, wi_below, wo_n)    == RGB{Float32}(0,0,0)
            @test evaluate(m, wo_n,    wo_below) == RGB{Float32}(0,0,0)
            @test evaluate(m, wi_below, wo_below) == RGB{Float32}(0,0,0)
        end
    end

    @testset "Lambertian: f = albedo/pi exactly" begin
        alb = RGB{Float32}(0.8f0, 0.4f0, 0.2f0)
        m   = LambertianModel(alb)
        c   = Float32(1/pi)
        for wi in HEMISPHERE_DIRS, wo in HEMISPHERE_DIRS
            f = evaluate(m, wi, wo)
            @test f.r ≈ alb.r * c  atol=1e-7
            @test f.g ≈ alb.g * c  atol=1e-7
            @test f.b ≈ alb.b * c  atol=1e-7
        end
    end

    @testset "Reciprocity f(wi,wo) == f(wo,wi)" begin
        for m in ALL_MODELS
            name = nameof(typeof(m))
            for (wi, wo) in RECIPROCITY_PAIRS
                fab = evaluate(m, wi, wo)
                fba = evaluate(m, wo, wi)
                @test fab.r ≈ fba.r  atol=1e-5  rtol=1e-4
                @test fab.g ≈ fba.g  atol=1e-5  rtol=1e-4
                @test fab.b ≈ fba.b  atol=1e-5  rtol=1e-4
            end
        end
    end

    @testset "CookTorranceModel(albedo=0) == CookTorranceSpecularModel" begin
        r  = 0.4f0
        F0 = RGB{Float32}(0.04f0,0.04f0,0.04f0)
        full = CookTorranceModel(r, F0, RGB{Float32}(0,0,0))
        spec = CookTorranceSpecularModel(r, F0)
        for wi in HEMISPHERE_DIRS[1:4:end], wo in HEMISPHERE_DIRS[1:4:end]
            fa = evaluate(full, wi, wo)
            fb = evaluate(spec, wi, wo)
            @test fa.r ≈ fb.r  atol=1e-5
            @test fa.g ≈ fb.g  atol=1e-5
            @test fa.b ≈ fb.b  atol=1e-5
        end
    end

    @testset "GGXModel(albedo=0) == GGXSpecularModel" begin
        r  = 0.35f0
        F0 = RGB{Float32}(0.1f0,0.2f0,0.3f0)
        full = GGXModel(r, F0, RGB{Float32}(0,0,0))
        spec = GGXSpecularModel(r, F0)
        for wi in HEMISPHERE_DIRS[1:4:end], wo in HEMISPHERE_DIRS[1:4:end]
            fa = evaluate(full, wi, wo)
            fb = evaluate(spec, wi, wo)
            @test fa.r ≈ fb.r  atol=1e-5
            @test fa.g ≈ fb.g  atol=1e-5
            @test fa.b ≈ fb.b  atol=1e-5
        end
    end

    @testset "ScaledDiffuse(kd=1) == plain diffuse model" begin
        r   = 0.4f0
        F0  = RGB{Float32}(0.04f0,0.04f0,0.04f0)
        alb = RGB{Float32}(0.5f0,0.3f0,0.1f0)
        dirs = [sph(0.3f0,0.5f0), sph(0.8f0,2.0f0), wo_n, wo_30]
        for (A, B) in [
            (CookTorranceModel(r,F0,alb),       CookTorranceScaledDiffuseModel(r,F0,alb,1.0f0)),
            (GGXModel(r,F0,alb),                GGXScaledDiffuseModel(r,F0,alb,1.0f0)),
        ]
            for wi in dirs, wo in dirs
                fa = evaluate(A, wi, wo)
                fb = evaluate(B, wi, wo)
                @test fa.r ≈ fb.r  atol=1e-5
                @test fa.g ≈ fb.g  atol=1e-5
                @test fa.b ≈ fb.b  atol=1e-5
            end
        end
    end

    @testset "ScaledDiffuse(kd=0) == specular-only model" begin
        r   = 0.4f0
        F0  = RGB{Float32}(0.04f0,0.04f0,0.04f0)
        alb = RGB{Float32}(0.5f0,0.3f0,0.1f0)
        dirs = [sph(0.3f0,0.5f0), sph(0.8f0,2.0f0), wo_n, wo_30]
        for (A, B) in [
            (CookTorranceSpecularModel(r,F0),   CookTorranceScaledDiffuseModel(r,F0,alb,0.0f0)),
            (GGXSpecularModel(r,F0),            GGXScaledDiffuseModel(r,F0,alb,0.0f0)),
        ]
            for wi in dirs, wo in dirs
                fa = evaluate(A, wi, wo)
                fb = evaluate(B, wi, wo)
                @test fa.r ≈ fb.r  atol=1e-5
                @test fa.g ≈ fb.g  atol=1e-5
                @test fa.b ≈ fb.b  atol=1e-5
            end
        end
    end

    @testset "CookTorrance == GGX (same NDF/G/F, same params)" begin
        # Both use identical GGX NDF + Smith G + Schlick F, so for same params
        # the specular lobe must be numerically identical.
        r  = 0.5f0
        F0 = RGB{Float32}(0.04f0,0.04f0,0.04f0)
        ct = CookTorranceSpecularModel(r, F0)
        gx = GGXSpecularModel(r, F0)
        for wi in HEMISPHERE_DIRS[1:3:end], wo in HEMISPHERE_DIRS[1:3:end]
            fa = evaluate(ct, wi, wo)
            fb = evaluate(gx, wi, wo)
            @test fa.r ≈ fb.r  atol=1e-5
            @test fa.g ≈ fb.g  atol=1e-5
            @test fa.b ≈ fb.b  atol=1e-5
        end
    end

    @testset "Energy conservation: Lambertian reflectance = albedo" begin
        m   = LambertianModel(RGB{Float32}(0.8f0, 0.6f0, 0.4f0))
        ref = mc_integral(m, wo_n; N=200_000)
        @test ref.r ≈ 0.8  atol=0.02
        @test ref.g ≈ 0.6  atol=0.02
        @test ref.b ≈ 0.4  atol=0.02
    end

    @testset "GLSL shaders: readable, contain brdf_evaluate, uniforms consistent" begin
        for m in ALL_MODELS
            code, declared_uniforms = glsl_shader(m)
            @test length(code) > 50
            @test occursin("brdf_evaluate", code)
            @test occursin("uniform", code)
            @test length(declared_uniforms) > 0
            params = shader_parameters(m)
            for u in declared_uniforms
                @test haskey(params, u)
            end
        end
    end

end
