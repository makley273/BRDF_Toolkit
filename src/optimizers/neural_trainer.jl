"""
Neural BRDF trainer — fits a `NeuralBRDFModel` (MLP) to measured data.

Trains a small MLP with Lux.jl + Optimisers.jl (Adam / AdamW, Zygote autodiff)
to reproduce the measured BRDF, then extracts the trained weights into the plain
`Float32` `NeuralBRDFModel` for fast evaluation and rendering.

Sampling, cosine weighting, and the loss transform are reused from
`_fit_common.jl`, so the neural fit honours the same `:sampling` / `:loss` /
`:cosine_weight` knobs as the analytical optimizers. `loss = :log` with
`cosine_weight = true` is the Sztrajman-style log-space loss on cosine-weighted
reflectance, with an ε-floored relative log rather than their log1p mapping:
log1p is effectively linear for the dark hemisphere (BRDF values ≪ 1), which
lets the specular peak drag the diffuse floor to zero — nets wider than ~40
trained to black materials. See `_residual_function`.

The network input is the reciprocity-enforcing mapping `Θ` built by the model's
`_recip_angles`, so the fitted BRDF satisfies Helmholtz reciprocity exactly.

# Usage
```julia
data  = load_brdf("path/to/material.binary")
model = fit_brdf(NeuralBRDFTrainer(; hidden=[21,21], epochs=150), NeuralBRDFModel, data)
```
"""

using Lux
using LuxCUDA          # CUDA backend + `gpu_device()` detection (falls back to CPU)
using Optimisers
using Zygote
using Random
using StaticArrays
using LinearAlgebra

# ============================================================================
# Optimizer Type
# ============================================================================

"""
    NeuralBRDFTrainer <: BRDFOptimizer

Adam/AdamW trainer for `NeuralBRDFModel`.

# Fields
- `hidden`:        Hidden-layer widths, e.g. `[21, 21]` (Sztrajman-like default).
- `hidden_act`:    Hidden activation (`:relu`, `:gelu`, `:tanh`, `:softplus`).
- `out_act`:       Non-negativity output activation (`:softplus` default, `:exp`,
                   `:square`, `:abs`, `:relu`).
- `optimizer`:     `:adam` or `:adamw` (decoupled weight decay).
- `learning_rate`: Adam learning rate.
- `weight_decay`:  AdamW weight decay (ignored for `:adam`).
- `epochs`:        Training epochs.
- `batch_size`:    Minibatch size.
- `n_samples`:     Number of `(wi, wo)` training pairs.
- `sampling`:      Direction sampling (`:rusinkiewicz`, `:uniform_half`, `:mixed`,
                   `:uniform`); shared with the analytical optimizers.
- `loss`:          Loss transform (`:log`, `:linear`, `:cuberoot`, `:ngan`,
                   `:huber`, `:logcosh`); shared via `_residual_function`.
- `cosine_weight`: Weight residuals by `cos(θi)` (matches Sztrajman with `:log`).
- `seed`:          RNG seed for reproducible weight initialisation / shuffling.
- `anisotropic`:   Use the anisotropic 6-input feature mapping (keeps `φ_h`) so
                   brushed/anisotropic materials can be fit. `false` (default) uses
                   the isotropic 4-input mapping — fewer params, less data, and
                   correct for isotropic materials (e.g. MERL). Both are exactly
                   reciprocity-invariant. See `_brdf_features`.
- `use_gpu`:       Train on a CUDA GPU when one is functional. Falls back to the
                   CPU (with a warning) if no GPU is available, so it never
                   crashes. Worth enabling only for wide/deep nets and large
                   `n_samples` / `batch_size`; tiny nets are faster on CPU.
"""
struct NeuralBRDFTrainer <: BRDFOptimizer
    hidden::Vector{Int}
    hidden_act::Symbol
    out_act::Symbol
    optimizer::Symbol
    learning_rate::Float64
    weight_decay::Float64
    epochs::Int
    batch_size::Int
    n_samples::Int
    sampling::Symbol
    loss::Symbol
    cosine_weight::Bool
    seed::Int
    anisotropic::Bool
    use_gpu::Bool
end

"""
    _parse_hidden(spec) -> Vector{Int}

Accept hidden-layer widths as a `Vector{<:Integer}` or a comma-separated string
(`"21,21"`, the form supplied by the GUI text field). Empty / unparsable input
falls back to a single 21-wide layer.
"""
function _parse_hidden(spec)
    if spec isa AbstractVector
        return Int[Int(x) for x in spec]
    elseif spec isa AbstractString
        parts = filter(!isempty, strip.(split(spec, ',')))
        widths = Int[]
        for p in parts
            v = tryparse(Int, p)
            (v === nothing || v <= 0) && continue
            push!(widths, v)
        end
        return isempty(widths) ? Int[21] : widths
    else
        throw(ArgumentError("hidden must be a Vector{Int} or comma-separated String, got $(typeof(spec))"))
    end
end

function NeuralBRDFTrainer(;
    hidden = [21, 21],
    hidden_act::Symbol = :relu,
    out_act::Symbol = :softplus,
    optimizer::Symbol = :adam,
    learning_rate::Real = 1e-3,
    weight_decay::Real = 1e-4,
    epochs::Integer = 150,
    batch_size::Integer = 512,
    n_samples::Integer = 8192,
    sampling::Symbol = :rusinkiewicz,
    loss::Symbol = :log,
    cosine_weight::Bool = true,
    seed::Integer = 1,
    anisotropic::Bool = false,
    use_gpu::Bool = false,
)
    hidden_act in NEURAL_HIDDEN_ACTS ||
        throw(ArgumentError("hidden_act must be one of $(NEURAL_HIDDEN_ACTS), got $hidden_act"))
    out_act in NEURAL_OUTPUT_ACTS ||
        throw(ArgumentError("out_act must be one of $(NEURAL_OUTPUT_ACTS), got $out_act"))
    optimizer in (:adam, :adamw) ||
        throw(ArgumentError("optimizer must be :adam or :adamw, got $optimizer"))
    loss in (:log, :linear, :cuberoot, :ngan, :huber, :logcosh) ||
        throw(ArgumentError("loss must be :log, :linear, :cuberoot, :ngan, :huber, or :logcosh"))
    sampling in VALID_SAMPLINGS ||
        throw(ArgumentError("sampling must be one of $(VALID_SAMPLINGS); got $sampling"))
    epochs >= 1 || throw(ArgumentError("epochs must be >= 1, got $epochs"))
    batch_size >= 1 || throw(ArgumentError("batch_size must be >= 1, got $batch_size"))
    n_samples >= 1 || throw(ArgumentError("n_samples must be >= 1, got $n_samples"))

    return NeuralBRDFTrainer(_parse_hidden(hidden), hidden_act, out_act, optimizer,
                             Float64(learning_rate), Float64(weight_decay), Int(epochs),
                             Int(batch_size), Int(n_samples), sampling, loss,
                             cosine_weight, Int(seed), anisotropic, use_gpu)
end

# ============================================================================
# Lux network construction (uses the model's own activations for exact match)
# ============================================================================

"""
    _build_chain(arch, hidden_act, out_act) -> Lux.Chain

Build a `Dense` MLP `arch[1] → arch[2] → … → arch[end]` with `hidden_act` on the
hidden layers and `out_act` on the final layer.

The Dense layers use the model's own concrete scalar activations
(`_hidden_activation` / `_output_activation`), so the network computes byte-for-byte
the same function as `NeuralBRDFModel`'s manual forward pass. Those are singleton
(isbits) functions rather than `Symbol`-capturing closures, so the chain also
compiles into CUDA kernels for GPU training.
"""
function _build_chain(arch::Vector{Int}, hidden_act::Symbol, out_act::Symbol)
    n_layers = length(arch) - 1
    layers = Vector{Any}(undef, n_layers)
    for i in 1:n_layers
        act = i < n_layers ? _hidden_activation(hidden_act) : _output_activation(out_act)
        layers[i] = Lux.Dense(arch[i] => arch[i+1], act)
    end
    return Lux.Chain(layers...)
end

# ============================================================================
# Fit Driver
# ============================================================================

"""
    fit_brdf(opt::NeuralBRDFTrainer, ModelType, data) -> NeuralBRDFModel

Train an MLP to reproduce `data`. Only `NeuralBRDFModel` is supported.
"""
function fit_brdf(opt::NeuralBRDFTrainer,
                  ::Type{M},
                  data::MeasuredBRDF) where {M <: BRDFModel}
    M === NeuralBRDFModel ||
        throw(ArgumentError("NeuralBRDFTrainer can only fit NeuralBRDFModel, not $M"))

    rng = Random.MersenneTwister(opt.seed)

    # --- Build the training set (reuse shared sampling) ----------------------
    wis, wos = _sample_directions(opt.sampling, opt.n_samples)
    N = length(wis)

    n_in = _feature_dim(opt.anisotropic)    # 4 (isotropic) or 6 (anisotropic)
    X = Matrix{Float32}(undef, n_in, N)     # reciprocity-invariant input features
    Y = Matrix{Float32}(undef, 3, N)        # raw measured RGB targets
    w = Vector{Float32}(undef, N)           # per-sample cosine weight
    @inbounds for k in 1:N
        X[:, k] = _brdf_features(opt.anisotropic, wis[k], wos[k])
        c = lookup(data, wis[k], wos[k])
        Y[1, k] = Float32(c.r); Y[2, k] = Float32(c.g); Y[3, k] = Float32(c.b)
        w[k] = opt.cosine_weight ? Float32(wis[k][3]) : 1.0f0
    end

    # --- Pick the compute device --------------------------------------------
    # `gpu_device()` returns a CUDA device when one is functional and falls back
    # to a `CPUDevice` otherwise; we detect that fallback by comparing against a
    # fresh `cpu_device()` so an unavailable GPU degrades gracefully (no crash).
    dev = opt.use_gpu ? gpu_device() : cpu_device()
    if opt.use_gpu && dev isa typeof(cpu_device())
        @warn "NeuralBRDFTrainer: use_gpu=true but no functional CUDA GPU found; training on CPU."
    end

    # Move the training set onto the device once (no per-batch host↔device copies).
    Xd = dev(X); Yd = dev(Y); wd = dev(w)

    # --- Build & initialise the Lux network ---------------------------------
    arch  = vcat(n_in, opt.hidden, 3)
    model = _build_chain(arch, opt.hidden_act, opt.out_act)
    ps, st = Lux.setup(rng, model)
    ps = dev(ps); st = dev(st)

    rule = opt.optimizer === :adamw ?
        Optimisers.AdamW(opt.learning_rate, (0.9, 0.999), opt.weight_decay) :
        Optimisers.Adam(opt.learning_rate)
    train_state = Lux.Training.TrainState(model, ps, st, rule)

    residual_fn = _residual_function(opt.loss)

    # Loss over a minibatch: weight predictions and targets identically (matches
    # `_fit_predict` / `_fit_targets`), then apply the shared residual transform.
    function _loss_fn(model, ps, st, batch)
        Xb, Yb, wb = batch
        Yhat, st_new = model(Xb, ps, st)
        wrow = reshape(wb, 1, :)
        R = residual_fn.(Yhat .* wrow, Yb .* wrow)
        return sum(abs2, R) / length(R), st_new, (;)
    end

    # --- Training loop -------------------------------------------------------
    bs = min(opt.batch_size, N)
    idx = collect(1:N)
    backend = AutoZygote()
    for _ in 1:opt.epochs
        Random.shuffle!(rng, idx)
        for s in 1:bs:N
            cols = idx[s:min(s + bs - 1, N)]
            batch = (Xd[:, cols], Yd[:, cols], wd[cols])
            _, _, _, train_state = Lux.Training.single_train_step!(
                backend, _loss_fn, batch, train_state)
        end
    end

    # --- Extract trained weights into the plain-array model ------------------
    final_ps = train_state.parameters
    n_layers = length(arch) - 1
    Ws = Vector{Matrix{Float32}}(undef, n_layers)
    bs_vec = Vector{Vector{Float32}}(undef, n_layers)
    for i in 1:n_layers
        layer = getproperty(final_ps, Symbol("layer_", i))
        # `Array(...)` first pulls the weights back from the GPU (a no-op on CPU)
        # so the stored `NeuralBRDFModel` always holds plain host `Float32` arrays.
        Ws[i] = Matrix{Float32}(Array(layer.weight))
        bs_vec[i] = Vector{Float32}(vec(Array(layer.bias)))
    end

    return NeuralBRDFModel(Ws, bs_vec, opt.hidden_act, opt.out_act, collect(arch),
                           opt.anisotropic)
end

# ============================================================================
# Option declaration (introspected by the UI)
# ============================================================================

optimizer_options(::Type{NeuralBRDFTrainer}) = OptimizerOption[
    OptimizerOption(:hidden, "Hidden layers", :text, "21,21",
                    Any[], (0, 0), nothing,
                    "Comma-separated hidden layer widths, e.g. 21,21 or 64,64,32"),
    OptimizerOption(:hidden_act, "Hidden act", :choice, :relu,
                    Any[NEURAL_HIDDEN_ACTS...], (0, 0), nothing,
                    "Hidden-layer activation"),
    OptimizerOption(:out_act, "Output act", :choice, :softplus,
                    Any[NEURAL_OUTPUT_ACTS...], (0, 0), nothing,
                    "Non-negativity output activation"),
    OptimizerOption(:optimizer, "Optimizer", :choice, :adam,
                    Any[:adam, :adamw], (0, 0), nothing,
                    "Gradient optimizer (AdamW adds decoupled weight decay)"),
    OptimizerOption(:learning_rate, "Learn rate", :float, 1e-3,
                    Any[], (1e-5, 1e-1), 1e-4,
                    "Adam learning rate"),
    OptimizerOption(:weight_decay, "Weight decay", :float, 1e-4,
                    Any[], (0.0, 1e-1), 1e-4,
                    "AdamW weight decay (ignored for Adam)"),
    OptimizerOption(:epochs, "Epochs", :int, 150,
                    Any[], (1, 2000), 10,
                    "Training epochs"),
    OptimizerOption(:batch_size, "Batch size", :int, 512,
                    Any[], (16, 8192), 16,
                    "Minibatch size"),
    OptimizerOption(:n_samples, "Samples", :int, 8192,
                    Any[], (256, 65536), 256,
                    "Number of (wi, wo) training pairs"),
    OptimizerOption(:sampling, "Sampling", :choice, :rusinkiewicz,
                    Any[VALID_SAMPLINGS...], (0, 0), nothing,
                    "Direction sampling strategy (:uniform barely samples θ_h≈0 — the " *
                    "unconstrained net loses the specular highlight; baseline only)"),
    OptimizerOption(:loss, "Loss", :choice, :log,
                    Any[:log, :linear, :cuberoot, :ngan, :huber, :logcosh], (0, 0), nothing,
                    "Loss transform (:log = ε-floored relative log; Sztrajman-style with cosine weight)"),
    OptimizerOption(:anisotropic, "Anisotropic", :bool, false,
                    Any[], (0, 0), nothing,
                    "Keep φ_h (6 inputs) so anisotropic materials can be fit; off = isotropic (4 inputs)"),
    OptimizerOption(:cosine_weight, "Cosine weight", :bool, true,
                    Any[], (0, 0), nothing,
                    "Weight residuals by cos(θi)"),
    OptimizerOption(:seed, "Seed", :int, 1,
                    Any[], (0, 100000), 1,
                    "RNG seed for reproducible training"),
    OptimizerOption(:use_gpu, "Use GPU", :bool, false,
                    Any[], (0, 0), nothing,
                    "Train on a CUDA GPU if available (falls back to CPU)"),
]

# ============================================================================
# Registration
# ============================================================================

export NeuralBRDFTrainer, fit_brdf
register_optimizer!(NeuralBRDFTrainer)
