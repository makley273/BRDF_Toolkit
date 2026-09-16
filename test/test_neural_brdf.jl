"""
Tests for the neural BRDF model (`NeuralBRDFModel`) and its trainer
(`NeuralBRDFTrainer`).

Covers:
- Reciprocity by construction (structural — holds for any weights).
- Non-negativity for every output activation.
- Forward-pass / weight-extraction consistency against the actual Lux network.
- A short smoke fit on synthetic constant data (loss decreases, output sensible).
- Architecture-string parsing for the GUI `:text` option.
- The trainer rejects non-neural model types.
"""

using Test
using BRDFPlatform
using StaticArrays
using Colors
using LinearAlgebra
using Random
using Lux

const BP = BRDFPlatform

# Pull in the internals under test (not part of the public API).
using BRDFPlatform: NeuralBRDFModel, NeuralBRDFTrainer, _brdf_features, _forward,
                    _feature_dim, _parse_hidden, _build_chain, NEURAL_OUTPUT_ACTS

# --- helpers ----------------------------------------------------------------

# Random (untrained) model with the given architecture. `anisotropic` is inferred
# from the input width (6 → anisotropic, else isotropic) but can be overridden.
function _random_model(arch::Vector{Int}; seed = 1, hidden_act = :relu,
                       out_act = :softplus, anisotropic = (arch[1] == 6))
    rng = MersenneTwister(seed)
    Ws = [randn(rng, Float32, arch[i+1], arch[i]) for i in 1:length(arch)-1]
    bs = [randn(rng, Float32, arch[i+1])          for i in 1:length(arch)-1]
    return NeuralBRDFModel(Ws, bs, hidden_act, out_act, arch, anisotropic)
end

# Random upper-hemisphere unit direction.
function _rand_dir(rng)
    z = rand(rng, Float32) * 0.98f0 + 0.01f0          # keep away from the horizon
    r = sqrt(max(0f0, 1f0 - z^2))
    phi = 2f0 * Float32(π) * rand(rng, Float32)
    return SVector{3,Float32}(r * cos(phi), r * sin(phi), z)
end

# Synthetic constant measured BRDF for the smoke fit.
struct _ConstData
    v::Float32
end
BRDFPlatform._brdf_value(d::_ConstData, ::Vector{Float32},
                         ::SVector{3,Float32}, ::SVector{3,Float32}) =
    RGB{Float32}(d.v, d.v, d.v)

# Synthetic *anisotropic* BRDF: a half-vector lobe stretched along the tangent (x)
# vs bitangent (y), so the value genuinely depends on the half-vector azimuth φ_h.
struct _AnisoData end
function BRDFPlatform._brdf_value(::_AnisoData, ::Vector{Float32},
                                  wi::SVector{3,Float32}, wo::SVector{3,Float32})
    h = normalize(wi + wo)
    v = 0.05f0 + 0.6f0 * exp(-((h[1] / 0.35f0)^2 + (h[2] / 0.08f0)^2))
    return RGB{Float32}(v, v, v)
end

@testset "NeuralBRDF" begin

    @testset "reciprocity by construction ($mode)" for (mode, arch) in
            (("isotropic", [4, 12, 8, 3]), ("anisotropic", [6, 12, 8, 3]))
        m = _random_model(arch; seed = 7)
        rng = MersenneTwister(123)
        for _ in 1:200
            wi = _rand_dir(rng); wo = _rand_dir(rng)
            a = evaluate(m, wi, wo)
            b = evaluate(m, wo, wi)
            # Bit-identical features ⇒ bit-identical output: reciprocity is exact.
            @test a.r == b.r
            @test a.g == b.g
            @test a.b == b.b
        end
    end

    @testset "feature mapping is wi↔wo bit-identical ($mode)" for (mode, aniso, dim) in
            (("isotropic", false, 4), ("anisotropic", true, 6))
        rng = MersenneTwister(5)
        for _ in 1:200
            wi = _rand_dir(rng); wo = _rand_dir(rng)
            f1 = _brdf_features(aniso, wi, wo)
            f2 = _brdf_features(aniso, wo, wi)
            @test length(f1) == dim == _feature_dim(aniso)
            @test f1 == f2          # exact bit-for-bit, not just ≈ (see docstring)
        end
    end

    @testset "isotropic features ignore φ_h, anisotropic features see it" begin
        # Rotate a (wi, wo) pair about the surface normal: isotropic features must
        # be invariant (φ_h dropped); anisotropic features must change (φ_h kept).
        rng = MersenneTwister(17)
        rotz(φ) = SMatrix{3,3,Float32}(cos(φ), sin(φ), 0, -sin(φ), cos(φ), 0, 0, 0, 1)
        for _ in 1:50
            wi = _rand_dir(rng); wo = _rand_dir(rng)
            R = rotz(1.0f0)                       # 1 rad about z
            wir = SVector{3,Float32}(R * wi); wor = SVector{3,Float32}(R * wo)
            @test _brdf_features(false, wi, wo) ≈ _brdf_features(false, wir, wor) atol = 1e-4
            # Off the specular peak the half-vector azimuth genuinely moves.
            if acos(clamp(normalize(wi + wo)[3], -1f0, 1f0)) > 0.15f0
                @test !isapprox(_brdf_features(true, wi, wo),
                                _brdf_features(true, wir, wor); atol = 1e-3)
            end
        end
    end

    @testset "non-negativity (all output activations)" begin
        rng = MersenneTwister(2)
        for oa in NEURAL_OUTPUT_ACTS
            m = _random_model([4, 16, 16, 3]; seed = 9, out_act = oa)
            for _ in 1:100
                wi = _rand_dir(rng); wo = _rand_dir(rng)
                c = evaluate(m, wi, wo)
                @test c.r >= 0 && c.g >= 0 && c.b >= 0
            end
        end
    end

    @testset "below-horizon returns zero" begin
        m = _random_model([4, 8, 3])
        below = SVector{3,Float32}(0.3, 0.2, -0.9)
        up    = SVector{3,Float32}(0.0, 0.0, 1.0)
        c = evaluate(m, below, up)
        @test c.r == 0 && c.g == 0 && c.b == 0
    end

    @testset "forward pass matches Lux network (extraction)" begin
        arch = [4, 8, 5, 3]
        model = _build_chain(arch, :relu, :softplus)
        ps, st = Lux.setup(MersenneTwister(0), model)
        Ws = [Matrix{Float32}(getproperty(ps, Symbol("layer_", i)).weight) for i in 1:length(arch)-1]
        bs = [Vector{Float32}(vec(getproperty(ps, Symbol("layer_", i)).bias)) for i in 1:length(arch)-1]
        nm = NeuralBRDFModel(Ws, bs, :relu, :softplus, arch)

        rng = MersenneTwister(11)
        for _ in 1:50
            θ = rand(rng, Float32, 4)
            y_lux, _ = model(reshape(θ, 4, 1), ps, st)
            y_man = _forward(nm, θ)
            @test vec(y_lux) ≈ y_man rtol = 1e-4
        end
    end

    @testset "architecture parsing" begin
        @test _parse_hidden("32,16,8") == [32, 16, 8]
        @test _parse_hidden(" 21 , 21 ") == [21, 21]
        @test _parse_hidden([64, 32]) == [64, 32]
        @test _parse_hidden("") == [21]          # fallback
        @test _parse_hidden("bad,,3") == [3]     # skips unparsable
    end

    @testset "trainer rejects non-neural models" begin
        data = MeasuredBRDF(_ConstData(0.1f0), Float32[630, 532, 465], "const";
                            source_format = :synthetic)
        @test_throws ArgumentError fit_brdf(NeuralBRDFTrainer(), LambertianModel, data)
    end

    @testset "smoke fit on constant data" begin
        target = 0.1f0
        data = MeasuredBRDF(_ConstData(target), Float32[630, 532, 465], "const";
                            source_format = :synthetic)
        opt = NeuralBRDFTrainer(; hidden = [16, 16], epochs = 60, n_samples = 1024,
                                batch_size = 256, learning_rate = 1e-3,
                                loss = :linear, cosine_weight = false, seed = 3)
        m = fit_brdf(opt, NeuralBRDFModel, data)

        @test m isa NeuralBRDFModel
        @test m.arch == [4, 16, 16, 3]

        # The fitted constant material should be finite, non-negative, and close
        # to the target across interior directions.
        rng = MersenneTwister(99)
        errs = Float64[]
        for _ in 1:200
            wi = _rand_dir(rng); wo = _rand_dir(rng)
            c = evaluate(m, wi, wo)
            @test isfinite(c.r) && isfinite(c.g) && isfinite(c.b)
            @test c.r >= 0
            push!(errs, abs(c.r - target))
        end
        @test sum(errs) / length(errs) < 0.05    # learned the constant (smoke-level)
    end

    @testset ":log residual keeps gradients in the dark hemisphere" begin
        # Regression: :log used to be log1p, which is ≈ linear for BRDF values
        # ≪ 1 — dark regions got vanishing loss, so wide nets fit the specular
        # peak and collapsed the diffuse floor to 0 (black render, NaN metrics).
        # The ε-floored relative log must penalize "predict 0 for a dark target"
        # on the log scale, not the linear one.
        r = BRDFPlatform._residual_function(:log)
        dark = r(0.0, 0.02)               # predict black where target is 0.02
        @test abs(dark) > 2.0             # log(1e-3/0.021) ≈ -3.0 (log1p gave -0.02)
        @test r(0.02, 0.02) ≈ 0.0 atol = 1e-12
        # Peak still compressed: a 1000× error at the peak stays O(log).
        @test abs(r(1000.0, 100.0)) < 3.0
        # Sign convention: under-prediction is negative, over-prediction positive.
        @test r(0.001, 0.1) < 0 < r(0.1, 0.001)
    end

    @testset ":cuberoot residual has a bounded slope near zero" begin
        # Regression: bare cbrt(pred) is finite-valued but its derivative
        # 1/(3·cbrt(pred)²) → +Inf as pred → 0⁺. A neural net's positive output
        # activation emits tiny-but-nonzero values across the dark hemisphere, so
        # Zygote/Adam multiplied in gradients up to ~1e7 and the weights NaN'd
        # (metals like weta_brushed_steel_satin, ≈0 over most of the hemisphere,
        # trigger it immediately). The cbrt_eps floor caps the slope; it must
        # stay symmetric so the residual still vanishes at pred == target.
        r = BRDFPlatform._residual_function(:cuberoot)
        @test r(0.02, 0.02) ≈ 0.0 atol = 1e-12          # vanishes at pred == target
        # Secant slope over a tiny positive prediction step is bounded (it is
        # ~1e6 for bare cbrt, ~30 with the 1e-5 floor).
        slope = (r(1e-10, 0.02) - r(0.0, 0.02)) / 1e-10
        @test isfinite(slope) && abs(slope) < 1e3
        # Still a genuine compressing transform away from zero.
        @test r(0.001, 0.1) < 0 < r(0.1, 0.001)         # sign convention
        @test abs(r(1000.0, 100.0)) < 8.0               # peak stays O(cbrt)
    end

    @testset ":logcosh residual is Float32-safe and stable" begin
        # Regression: cosh(d) == 1f0 exactly for |d| ≲ 5e-4 in Float32, so the
        # naive log(cosh(d)) returned R = 0 with d ≠ 0 (gradient lost; the sqrt
        # at 0 has an infinite derivative → NaN weights in neural training).
        r = BRDFPlatform._residual_function(:logcosh)
        for d in (5.0f-5, 3.0f-4, 5.0f-3)          # inside the old dead zone
            v = r(0.5f0 + d, 0.5f0)
            @test isfinite(v) && v > 0
            @test v ≈ d / sqrt(2.0) rtol = 1e-3    # small-|d| limit R = d/√2
        end
        # Branch continuity at the 1e-2 guard and correctness beyond it.
        @test r(0.5 + 0.01 - 1e-9, 0.5) ≈ r(0.5 + 0.01 + 1e-9, 0.5) rtol = 1e-4
        @test r(1.5, 0.5)^2 ≈ log(cosh(1.0)) rtol = 1e-6
        # Large |d| must not overflow cosh (old code branched at 20; the stable
        # form log(cosh d) = |d| + log1p(e^{-2|d|}) − log2 covers everything).
        @test r(1.0e4, 0.0)^2 ≈ 1.0e4 - log(2.0) rtol = 1e-6
        @test isfinite(r(0.0f0, 3.0f4))
    end

    @testset ":huber residual continuity at delta" begin
        r = BRDFPlatform._residual_function(:huber)   # delta = 0.1
        @test r(0.5 + 0.1 - 1e-9, 0.5) ≈ r(0.5 + 0.1 + 1e-9, 0.5) rtol = 1e-6
        @test r(0.55, 0.5) ≈ 0.05 / sqrt(2.0)                  # quadratic branch
        @test r(1.5, 0.5)^2 ≈ 0.1 * (1.0 - 0.05) rtol = 1e-12  # damped branch
        @test r(0.5, 1.5) == -r(1.5, 0.5)                      # odd symmetry
    end

    @testset "use_gpu trains and returns a host model (CPU fallback ok)" begin
        # Requesting the GPU must never crash: on a CPU-only machine the trainer
        # warns and falls back, and either way the returned model holds plain host
        # Float32 arrays so evaluate/rendering work unchanged.
        target = 0.1f0
        data = MeasuredBRDF(_ConstData(target), Float32[630, 532, 465], "const";
                            source_format = :synthetic)
        opt = NeuralBRDFTrainer(; hidden = [16, 16], epochs = 40, n_samples = 512,
                                batch_size = 256, learning_rate = 1e-3,
                                loss = :linear, cosine_weight = false, seed = 3,
                                use_gpu = true)
        m = fit_brdf(opt, NeuralBRDFModel, data)

        @test m isa NeuralBRDFModel
        @test all(W -> W isa Matrix{Float32}, m.Ws)
        @test all(b -> b isa Vector{Float32}, m.bs)

        rng = MersenneTwister(7)
        for _ in 1:50
            wi = _rand_dir(rng); wo = _rand_dir(rng)
            c = evaluate(m, wi, wo)
            @test isfinite(c.r) && c.r >= 0
        end
    end

    @testset "anisotropic mode captures φ_h variation" begin
        data = MeasuredBRDF(_AnisoData(), Float32[630, 532, 465], "aniso";
                            source_format = :synthetic)
        common = (; hidden = [24, 24], epochs = 80, n_samples = 2048, batch_size = 512,
                  learning_rate = 2e-3, loss = :linear, cosine_weight = false, seed = 5)
        m_iso = fit_brdf(NeuralBRDFTrainer(; common..., anisotropic = false),
                         NeuralBRDFModel, data)
        m_ani = fit_brdf(NeuralBRDFTrainer(; common..., anisotropic = true),
                         NeuralBRDFModel, data)

        @test m_iso.arch[1] == 4 && m_iso.anisotropic == false
        @test m_ani.arch[1] == 6 && m_ani.anisotropic == true

        # Rotating a (wi, wo) pair about the normal changes the true anisotropic
        # BRDF. The anisotropic model tracks that change; the isotropic model is
        # rotation-invariant by construction and cannot (iso_var is exactly 0).
        rotz(φ) = SMatrix{3,3,Float32}(cos(φ), sin(φ), 0, -sin(φ), cos(φ), 0, 0, 0, 1)
        rng = MersenneTwister(31)
        iso_var = 0.0; ani_var = 0.0
        R = rotz(Float32(π) / 3)
        for _ in 1:300
            wi = _rand_dir(rng); wo = _rand_dir(rng)
            wir = SVector{3,Float32}(R * wi); wor = SVector{3,Float32}(R * wo)
            iso_var += abs(evaluate(m_iso, wi, wo).r - evaluate(m_iso, wir, wor).r)
            ani_var += abs(evaluate(m_ani, wi, wo).r - evaluate(m_ani, wir, wor).r)
        end
        iso_var /= 300; ani_var /= 300

        @test iso_var < 1e-5                 # isotropic: rotation-invariant
        # Absolute floor: comfortably above training noise yet below what every
        # sampling strategy reaches on this budget (rusinkiewicz/uniform/
        # fibonacci all land 0.006–0.009). The real teeth is the relative check
        # below — ani_var is ~10^6× iso_var, so the aniso model demonstrably
        # learned φ_h dependence the iso model is invariant to by construction.
        @test ani_var > 0.003                # anisotropic: learned real φ_h dependence
        @test ani_var > 100 * iso_var
    end
end
