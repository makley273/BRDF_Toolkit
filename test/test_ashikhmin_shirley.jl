"""
Tests for the Ashikhmin-Shirley anisotropic Phong BRDF model
(`AshikhminShirleyModel`).

Covers:
- Plugin registration.
- Below-horizon directions return zero.
- Helmholtz reciprocity f(wi, wo) == f(wo, wi) (structural — holds for any params).
- Non-negativity and finiteness over random directions.
- Isotropy: nu == nv is rotation-invariant about the normal; nu != nv is not.
- Energy sanity of the pure-diffuse (Rs = 0) limit.
- A smoke fit that recovers the reflectance of a known synthetic material.
"""

using Test
using BRDFPlatform
using StaticArrays
using Colors
using LinearAlgebra
using Random

const BP = BRDFPlatform
using BRDFPlatform: BRDFRGB

# Random upper-hemisphere unit direction (kept away from the grazing horizon).
function _rand_dir(rng)
    z = rand(rng, Float32) * 0.98f0 + 0.01f0
    r = sqrt(max(0f0, 1f0 - z^2))
    phi = 2f0 * Float32(π) * rand(rng, Float32)
    return SVector{3,Float32}(r * cos(phi), r * sin(phi), z)
end

# Synthetic measured BRDF backed by a known Ashikhmin-Shirley model, so a fit can
# be checked against ground truth.
struct _ASData
    m::AshikhminShirleyModel{Float64}
end
function BRDFPlatform._brdf_value(d::_ASData, ::Vector{Float32},
                                  wi::SVector{3,Float32}, wo::SVector{3,Float32})
    c = evaluate(d.m, wi, wo)
    return RGB{Float32}(c.r, c.g, c.b)
end

@testset "AshikhminShirley" begin

    @testset "registration" begin
        @test :ashikhminshirleymodel in BP.list_plugins()[:models]
    end

    @testset "below-horizon returns zero" begin
        m = AshikhminShirleyModel(100.0, 100.0, 0.05, 0.3)
        below = SVector{3,Float32}(0.3, 0.2, -0.9)
        up    = SVector{3,Float32}(0.0, 0.0, 1.0)
        c = evaluate(m, below, up)
        @test c.r == 0 && c.g == 0 && c.b == 0
        @test evaluate(m, up, below).r == 0
    end

    @testset "reciprocity f(wi,wo) == f(wo,wi)" begin
        m = AshikhminShirleyModel(300.0, 40.0, BRDFRGB(0.04, 0.06, 0.08),
                                              BRDFRGB(0.25, 0.30, 0.20))
        rng = MersenneTwister(1)
        for _ in 1:300
            wi = _rand_dir(rng); wo = _rand_dir(rng)
            a = evaluate(m, wi, wo)
            b = evaluate(m, wo, wi)
            @test a.r ≈ b.r rtol = 1e-4 atol = 1e-8
            @test a.g ≈ b.g rtol = 1e-4 atol = 1e-8
            @test a.b ≈ b.b rtol = 1e-4 atol = 1e-8
        end
    end

    @testset "non-negativity and finiteness" begin
        rng = MersenneTwister(2)
        for (nu, nv) in ((1.0, 1.0), (10.0, 500.0), (1000.0, 1000.0))
            m = AshikhminShirleyModel(nu, nv, 0.05, 0.4)
            for _ in 1:200
                wi = _rand_dir(rng); wo = _rand_dir(rng)
                c = evaluate(m, wi, wo)
                @test isfinite(c.r) && isfinite(c.g) && isfinite(c.b)
                @test c.r >= 0 && c.g >= 0 && c.b >= 0
            end
        end
    end

    @testset "isotropy: nu==nv rotation-invariant, nu!=nv is not" begin
        rotz(φ) = SMatrix{3,3,Float32}(cos(φ), sin(φ), 0, -sin(φ), cos(φ), 0, 0, 0, 1)
        R = rotz(Float32(π) / 3)
        rng = MersenneTwister(3)

        m_iso = AshikhminShirleyModel(200.0, 200.0, 0.05, 0.3)
        m_ani = AshikhminShirleyModel(600.0, 30.0, 0.05, 0.3)
        iso_var = 0.0; ani_var = 0.0
        for _ in 1:300
            wi = _rand_dir(rng); wo = _rand_dir(rng)
            wir = SVector{3,Float32}(R * wi); wor = SVector{3,Float32}(R * wo)
            iso_var += abs(evaluate(m_iso, wi, wo).r - evaluate(m_iso, wir, wor).r)
            ani_var += abs(evaluate(m_ani, wi, wo).r - evaluate(m_ani, wir, wor).r)
        end
        iso_var /= 300; ani_var /= 300
        @test iso_var < 1e-4          # isotropic highlight: rotation-invariant
        @test ani_var > 100 * iso_var # anisotropic highlight: depends on azimuth
    end

    @testset "diffuse term matches closed form at wi == wo" begin
        # At wi == wo the half-vector h == wi, so h·wi == 1 and the Schlick term
        # (1 − h·wi)^5 == 0. With Rs = 0 the specular contribution is therefore
        # exactly zero, isolating the diffuse closed form:
        #   ρd = 28 Rd/(23π) · (1 − (1 − z/2)^5)²   (fl == fv here since wi == wo).
        Rd = 0.5
        m = AshikhminShirleyModel(50.0, 50.0, 0.0, Rd)
        for z in (0.15f0, 0.4f0, 0.7f0, 0.99f0)
            w = SVector{3,Float32}(sqrt(1 - z^2), 0f0, z)
            fl = 1.0 - (1.0 - z / 2.0)^5
            expected = 28.0 * Rd / (23.0 * π) * fl * fl
            @test evaluate(m, w, w).r ≈ expected rtol = 1e-5
        end
    end

    @testset "smoke fit recovers a known material" begin
        truth = AshikhminShirleyModel(200.0, 200.0, BRDFRGB(0.05, 0.05, 0.05),
                                                    BRDFRGB(0.30, 0.20, 0.10))
        data = MeasuredBRDF(_ASData(truth), Float32[630, 532, 465], "as-synth";
                            source_format = :synthetic)
        fit = fit_brdf(LBFGSB(; n_samples = 1500, n_starts = 2), AshikhminShirleyModel, data)

        @test fit isa AshikhminShirleyModel

        # Parameters are not uniquely identifiable, so compare the reflectance the
        # two models produce at interior directions rather than the raw params.
        rng = MersenneTwister(5)
        num = 0.0; den = 0.0
        for _ in 1:400
            wi = _rand_dir(rng); wo = _rand_dir(rng)
            t = evaluate(truth, wi, wo)
            f = evaluate(fit, wi, wo)
            num += abs(f.r - t.r) + abs(f.g - t.g) + abs(f.b - t.b)
            den += t.r + t.g + t.b
        end
        @test num / den < 0.1        # mean relative reflectance error < 10%
    end
end
