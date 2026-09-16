"""
Tests for the universal LUT's half-angle sampling grid.

The Rusinkiewicz box (theta_h, theta_d, phi_d) is over-complete: large
theta_h + theta_d combinations reconstruct to a direction pair with wi or wo
below the horizon, which is not a realizable reflection. `generate_lut` pulls
those cells back onto the valid boundary via `_max_theta_d`, so no cell is ever
queried at an impossible geometry.
"""

using Test
using BRDFPlatform
using StaticArrays
using LinearAlgebra: dot, normalize
using Base64: base64decode

const BP = BRDFPlatform

@testset "Universal LUT half-angle sampling" begin

    half = Float32(π) / 2.0f0

    @testset "_max_theta_d matches the analytic boundary" begin
        # phi_d = 0 or π: the tangential offset lies in the incidence plane and
        # the constraint collapses to the familiar theta_h + theta_d ≤ π/2.
        for thd in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0), pd in (0.0f0, Float32(π))
            th = Float32(deg2rad(thd))
            @test rad2deg(BP._max_theta_d(th, pd; z_min = 0.0f0)) ≈ (90.0 - thd) atol = 1e-3
        end

        # phi_d = π/2: the offset is parallel to the surface, wi.z == wo.z, and
        # the whole theta_d range stays above the horizon.
        for thd in (0.0, 30.0, 60.0, 89.0)
            th = Float32(deg2rad(thd))
            @test BP._max_theta_d(th, Float32(π) / 2; z_min = 0.0f0) ≈ half atol = 1e-5
        end

        # The z_min margin only ever shrinks the valid range.
        for thd in (0.0, 20.0, 55.0, 80.0), pdd in (0.0, 40.0, 90.0, 140.0)
            th, pd = Float32(deg2rad(thd)), Float32(deg2rad(pdd))
            @test BP._max_theta_d(th, pd) <= BP._max_theta_d(th, pd; z_min = 0.0f0)
        end
    end

    @testset "clamped grid never leaves the hemisphere" begin
        # Exhaustive sweep over several grid shapes: every cell the bake loop
        # queries must reconstruct to a pair with both directions above (or, on
        # the fully degenerate theta_h = π/2 slice, exactly at) the horizon.
        for dims in ((32, 32, 32, 8), (16, 24, 40, 4), (64, 8, 8, 1))
            N_th, N_td, N_pd, N_ph = dims
            worst = Inf32
            clamped = 0
            for i_ph in 0:(N_ph - 1), i_pd in 0:(N_pd - 1),
                i_td in 0:(N_td - 1), i_th in 0:(N_th - 1)

                phi_h = (Float32(i_ph) / Float32(N_ph)) * (2.0f0 * Float32(π))
                phi_d = BP._angle_linear(i_pd, 2.0f0 * Float32(π), N_pd)
                th    = BP._angle_theta_h(i_th, N_th)
                td_g  = BP._angle_linear(i_td, half, N_td)
                td    = min(td_g, BP._max_theta_d(th, phi_d))
                td < td_g && (clamped += 1)

                wi, wo = BP._angles_to_dirs(th, phi_h, td, phi_d)
                worst = min(worst, wi[3], wo[3])
            end
            # Only float roundoff on the theta_h = π/2 slice may dip below zero,
            # and generate_lut zeroes that slice rather than querying it.
            @test worst > -1.0f-6
            # Sanity: a meaningful slice of the box really is out of bounds, so
            # the guard is doing work rather than silently matching the old path.
            @test clamped > 0.1 * N_th * N_td * N_pd * N_ph
        end
    end

    @testset "clamping preserves valid cells" begin
        # Cells inside the valid region must be untouched: same directions as an
        # unclamped reconstruction.
        for (thd, tdd, pdd) in ((0.0, 0.0, 0.0), (10.0, 20.0, 30.0),
                                (30.0, 45.0, 90.0), (20.0, 60.0, 170.0))
            th, td, pd = Float32.(deg2rad.((thd, tdd, pdd)))
            if td <= BP._max_theta_d(th, pd)
                wi, wo = BP._angles_to_dirs(th, 0.0f0, td, pd)
                @test wi[3] >= 0.0f0
                @test wo[3] >= 0.0f0
                # h bisects the pair and theta_d is the angle to it.
                h = normalize(wi + wo)
                @test acos(clamp(dot(wi, h), -1.0f0, 1.0f0)) ≈ td atol = 1e-4
                @test acos(clamp(h[3], -1.0f0, 1.0f0)) ≈ th atol = 1e-4
            end
        end
    end

    @testset "generate_lut bakes a finite, non-negative table" begin
        model = LambertianModel(0.6f0)
        b64, cfg = generate_lut(model; dims = (16, 16, 16, 2), dtype = :float32)

        @test cfg["dims"] == [16, 16, 16, 2]
        raw = reinterpret(Float32, base64decode(b64))
        @test length(raw) == 16 * 16 * 16 * 2 * 3
        @test all(isfinite, raw)
        @test all(>=(0.0f0), raw)

        # Lambertian is albedo/π everywhere a pair exists. With the clamp, only
        # the degenerate theta_h = π/2 slice may be zero, so the vast majority of
        # texels must carry the constant value (before the clamp, whole blocks of
        # the box collapsed to zero because evaluate rejects below-horizon pairs).
        expected = 0.6f0 / Float32(π)
        red = @view raw[1:3:end]
        @test count(v -> isapprox(v, expected; rtol = 1e-3), red) > 0.9 * length(red)
    end
end
