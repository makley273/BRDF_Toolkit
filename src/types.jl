"""
Core type definitions for the BRDF Platform.

This file defines the abstract types and core data structures used throughout
the platform. All plugins must implement interfaces defined here.
"""

using StaticArrays
using Colors

# ============================================================================
# BRDFRGB — lightweight RGB carrier compatible with any T <: Real
# ============================================================================

"""
    BRDFRGB{T <: Real}

Minimal RGB triplet for BRDF evaluation results.  Unlike `Colors.RGB{T}`,
which requires `T <: Fractional` (i.e. `AbstractFloat | FixedPoint`), this
type accepts any `T <: Real`.  This lets `evaluate` propagate `ForwardDiff.Dual`
values through without hitting ColorTypes.jl's type-parameter constraint.

Fields are named `.r`, `.g`, `.b` so existing code that accesses
`brdf_value.r` / `.g` / `.b` continues to work unchanged.

The rendering pipeline converts to `Float32` explicitly when needed.
"""
struct BRDFRGB{T <: Real}
    r::T
    g::T
    b::T
end

# ============================================================================
# Abstract Types - Plugin Interfaces
# ============================================================================

"""
    BRDFData

Abstract supertype for all BRDF data representations.
Includes both measured data and fitted analytical models.
"""
abstract type BRDFData end

"""
    BRDFModel <: BRDFData

Abstract supertype for analytical BRDF models.
Each model must implement:
- `evaluate(model, wi, wo)` - Compute the BRDF value (Julia)

Rendering goes through the universal 4D-LUT, which samples `evaluate` directly, so
a model needs no GLSL. `display_parameters(model)` (for the UI readout) has a
generic fallback and is optional to override. Optimizers call `_fit_spec(Model)`,
defined in the model's own file.
"""
abstract type BRDFModel <: BRDFData end

"""
    BRDFOptimizer

Abstract supertype for fitting algorithms.
Each optimizer must implement:
- `fit_brdf(optimizer, model_type, data)` - Fit model to data

Optimizers may also declare their tunable knobs via `optimizer_options`
so the UI (and future hyperparameter tuners) can introspect them.
"""
abstract type BRDFOptimizer end

"""
    OptimizerOption

Declarative description of one tunable optimizer knob. The UI reads these
and renders a control; a hyperparameter tuner can read the same description
to know the search space.

# Fields
- `name`:        Keyword argument name passed to the optimizer constructor.
- `label`:       Human-readable label for the UI.
- `kind`:        One of `:choice`, `:int`, `:float`, `:bool`, `:text`. The UI maps each
                 to an input element; `:text` is the free-form escape hatch (e.g. the
                 neural trainer's `hidden` layer spec, `"21,21"`).
- `default`:     Default value (matches the optimizer's own default).
- `choices`:     Allowed values for `:choice` kind; empty otherwise.
- `range`:       `(min, max)` for `:int` / `:float`; `(0, 0)` otherwise.
- `step`:        Step size for `:int` / `:float`; `nothing` otherwise.
- `description`: One-line explanation, suitable for a tooltip.
"""
struct OptimizerOption
    name::Symbol
    label::String
    kind::Symbol
    default::Any
    choices::Vector{Any}
    range::Tuple{Any,Any}
    step::Any
    description::String
end

"""
    optimizer_options(::Type{<:BRDFOptimizer}) -> Vector{OptimizerOption}

Return the list of tunable options for an optimizer type. Default is empty;
optimizers override this to expose their constructor kwargs.
"""
optimizer_options(::Type{<:BRDFOptimizer}) = OptimizerOption[]


# ============================================================================
# Core Data Structure - Universal BRDF Representation
# ============================================================================

"""
    MeasuredBRDF <: BRDFData

Universal representation for measured BRDF data.
All format loaders convert to this structure.

**IMPORTANT:** External code should NEVER access `_data` directly.
Use the `brdf_value` function to query BRDF values.

# Fields (Internal - DO NOT ACCESS DIRECTLY)
- `_data`: Internal storage (format-specific, opaque). Each format type
  (e.g. MERLData) encapsulates its own sampling knowledge internally.

# Fields (Public)
- `wavelengths`: Wavelengths in nanometers
- `name`: Material name
- `source_format`: Original format (:merl, :bird, :utia, etc.)
- `status`: Load status (:ok, :partial, :failed)
- `error_msg`: Error message if status != :ok

# Public Interface
Use `brdf_value(brdf, wi, wo)` to query BRDF values at specific directions.
Use `lookup_batch(brdf, wis, wos)` for efficient batch queries.
"""
struct MeasuredBRDF <: BRDFData
    _data::Any                          # Internal storage (opaque)
    wavelengths::Vector{Float32}        # nm
    name::String
    source_format::Symbol
    status::Symbol                      # :ok, :partial, :failed
    error_msg::Union{Nothing, String}
end

# Constructor with default values
function MeasuredBRDF(
    data::Any,
    wavelengths::Vector{Float32},
    name::String;
    source_format::Symbol = :unknown,
    status::Symbol = :ok,
    error_msg::Union{Nothing, String} = nothing
)
    MeasuredBRDF(data, wavelengths, name, source_format, status, error_msg)
end

# ============================================================================
# Lookup Interface - MUST be implemented by each sampling pattern
# ============================================================================

"""
    brdf_value(brdf::MeasuredBRDF, wi::SVector{3, Float32}, wo::SVector{3, Float32}) -> RGB{Float32}

Query BRDF value at specific incident and outgoing directions.

This is the PRIMARY interface for accessing measured BRDF data.
External code should NEVER access internal storage directly.

# Arguments
- `brdf`: MeasuredBRDF instance
- `wi`: Incident direction (normalized, pointing away from surface)
- `wo`: Outgoing direction (normalized, pointing away from surface)

# Returns
RGB{Float32} representing the BRDF value at these directions.
"""
function brdf_value(brdf::MeasuredBRDF, wi::SVector{3, T1}, wo::SVector{3, T2}) where {T1, T2}
    # Normalize inputs
    wi_norm = normalize_vec(SVector{3, Float32}(wi))
    wo_norm = normalize_vec(SVector{3, Float32}(wo))

    # Dispatch to format-specific implementation
    return _brdf_value(brdf._data, brdf.wavelengths, wi_norm, wo_norm)
end

# Convenience overload for tuples (accept mixed numeric types)
function brdf_value(brdf::MeasuredBRDF, wi::Tuple{<:Real,<:Real,<:Real}, wo::Tuple{<:Real,<:Real,<:Real})
    wi_vec = SVector{3, Float32}(wi[1], wi[2], wi[3])
    wo_vec = SVector{3, Float32}(wo[1], wo[2], wo[3])
    return brdf_value(brdf, wi_vec, wo_vec)
end

# ============================================================================
# Unified Lookup Interface — common denominator for analytical + measured models
# ============================================================================

"""
    lookup(brdf::BRDFData, wi::SVector{3,Float32}, wo::SVector{3,Float32}) -> RGB{Float32}

Unified BRDF evaluation interface for both measured data and analytical models.

This is the **universal entry point**: use `lookup` when you don't know (or don't
care) whether you have a `MeasuredBRDF` or a `BRDFModel`.

- For `MeasuredBRDF`: delegates to `brdf_value` (format-specific table lookup).
- For `BRDFModel`:    delegates to `evaluate` (analytical evaluation).
"""
function lookup(brdf::MeasuredBRDF, wi::SVector{3, T1}, wo::SVector{3, T2}) where {T1, T2}
    return brdf_value(brdf, wi, wo)
end

function lookup(model::BRDFModel, wi::SVector{3, T1}, wo::SVector{3, T2}) where {T1, T2}
    wi_f = SVector{3, Float32}(wi)
    wo_f = SVector{3, Float32}(wo)
    return evaluate(model, wi_f, wo_f)
end

# Convenience overloads for tuples
function lookup(brdf::BRDFData, wi::Tuple{<:Real,<:Real,<:Real}, wo::Tuple{<:Real,<:Real,<:Real})
    wi_vec = SVector{3, Float32}(wi[1], wi[2], wi[3])
    wo_vec = SVector{3, Float32}(wo[1], wo[2], wo[3])
    return lookup(brdf, wi_vec, wo_vec)
end

"""
    lookup_batch(brdf::MeasuredBRDF, wis, wos) -> Vector{RGB{Float32}}

Efficiently query BRDF values for multiple direction pairs.

# Arguments
- `brdf`: MeasuredBRDF instance
- `wis`: Vector of incident directions
- `wos`: Vector of outgoing directions (same length as wis)

# Returns
Vector of RGB{Float32} values, one per direction pair.

# Implementation
Default implementation calls brdf_value() for each pair.
Format-specific implementations can override for better performance.
"""
function lookup_batch(brdf::MeasuredBRDF,
                     wis::AbstractVector{<:SVector{3}},
                     wos::AbstractVector{<:SVector{3}})
    @assert length(wis) == length(wos) "wis and wos must have same length"

    # Check if format provides optimized batch lookup
    if hasmethod(lookup_batch, Tuple{typeof(brdf._data), typeof(brdf.wavelengths), typeof(wis), typeof(wos)})
        return lookup_batch(brdf._data, brdf.wavelengths, wis, wos)
    end

    # Default: loop over single lookups
    return [brdf_value(brdf, wi, wo) for (wi, wo) in zip(wis, wos)]
end

# ============================================================================
# Helper Functions
# ============================================================================

# ============================================================================
# BRDFModel Interface — declared here, implemented by each model plugin
# ============================================================================

"""
    evaluate(model::BRDFModel, wi, wo) -> BRDFRGB{T}

Compute the BRDF value for incident direction `wi` and outgoing direction `wo`.
Both vectors are normalized and point away from the surface.
The surface normal is assumed to be (0, 0, 1).

Must be implemented by every BRDFModel plugin.
"""
function evaluate end

"""
    display_parameters(model::BRDFModel) -> Dict{String, Any}

Return a mapping from parameter name to its current value, for the UI readout
panel. A model plugin does **not** need to implement this: the generic fallback
below reflects the struct's fields automatically (`BRDFRGB` fields become
`[r, g, b]` vectors). Define a method only to customize the readout (see
`NeuralBRDFModel`, which reports an architecture summary instead of raw weights).
"""
function display_parameters(model::BRDFModel)::Dict{String, Any}
    out = Dict{String, Any}()
    for f in fieldnames(typeof(model))
        v = getfield(model, f)
        out[String(f)] = v isa BRDFRGB ? Float32[v.r, v.g, v.b] : v
    end
    return out
end

# Deprecated alias: the model render path no longer uses GLSL (the universal
# 4D-LUT samples `evaluate` directly), so `shader_parameters` is a misnomer. Kept
# so any caller of the old name still resolves to the readout.
shader_parameters(model::BRDFModel) = display_parameters(model)

"""
    n_samples_str(data) -> String

Return a human-readable string describing the number of samples.
Format-specific types should override this.
"""
n_samples_str(::Any) = "unknown"

"""
    vec3(x, y, z) -> SVector{3, Float32}

Convenience constructor for 3D vectors.
"""
vec3(x, y, z) = SVector{3, Float32}(x, y, z)

"""
    normalize_vec(v) -> SVector{3, Float32}

Normalize a 3D vector.
"""
function normalize_vec(v::SVector{3, T}) where T
    len = sqrt(v[1]^2 + v[2]^2 + v[3]^2)
    return len > 0 ? v / len : v
end

"""
    rgb(r, g, b) -> RGB{Float32}

Convenience constructor for RGB colors.
"""
rgb(r, g, b) = RGB{Float32}(r, g, b)

# ============================================================================
# Plugin Registry (populated at runtime)
# ============================================================================

"""
Global registry for format plugins.
Populated by `register_format!()` calls.
"""
const FORMAT_REGISTRY = Dict{Symbol, Type{<:BRDFData}}()

"""
Global registry for model plugins.
Populated by `register_model!()` calls.
"""
const MODEL_REGISTRY = Dict{Symbol, Type{<:BRDFModel}}()

"""
Global registry for optimizer plugins.
Populated by `register_optimizer!()` calls.
"""
const OPTIMIZER_REGISTRY = Dict{Symbol, Type{<:BRDFOptimizer}}()

"""
    register_format!(format_type::Type{<:BRDFData})

Register a format plugin. Call this at the end of each format file.
"""
function register_format!(format_type::Type{<:BRDFData})
    name = Symbol(lowercase(string(nameof(format_type))))
    FORMAT_REGISTRY[name] = format_type
    Core.eval(@__MODULE__, Expr(:export, nameof(format_type)))
    @info "Registered format plugin: $name"
    return nothing
end

"""
    register_model!(model_type::Type{<:BRDFModel})

Register a model plugin. Call this at the end of each model file.
"""
function register_model!(model_type::Type{<:BRDFModel})
    name = Symbol(lowercase(string(nameof(model_type))))
    MODEL_REGISTRY[name] = model_type
    # Auto-export the model's type name so `using BRDFPlatform` brings it into
    # scope — a model plugin needs no manual `export` line, here or in its file.
    Core.eval(@__MODULE__, Expr(:export, nameof(model_type)))
    @info "Registered model plugin: $name"
    return nothing
end

"""
    register_optimizer!(optimizer_type::Type{<:BRDFOptimizer})

Register an optimizer plugin. Call this at the end of each optimizer file.
"""
function register_optimizer!(optimizer_type::Type{<:BRDFOptimizer})
    name = Symbol(lowercase(string(nameof(optimizer_type))))
    OPTIMIZER_REGISTRY[name] = optimizer_type
    Core.eval(@__MODULE__, Expr(:export, nameof(optimizer_type)))
    @info "Registered optimizer plugin: $name"
    return nothing
end

# ============================================================================
# Server networking helpers (shared by all run_*_app entry points)
# ============================================================================

import Sockets

"""
    _lan_ip_rank(s::String) -> Int

Preference score for an IPv4 string (higher = more likely the address other
devices on the *physical* LAN use). Ordinary private ranges beat CGNAT/VPN
(`100.64.0.0/10`, e.g. Tailscale) and link-local (`169.254.x`).
"""
function _lan_ip_rank(s::String)
    startswith(s, "192.168.") && return 4
    startswith(s, "10.")      && return 3
    # 172.16.0.0/12
    if startswith(s, "172.")
        oct2 = tryparse(Int, split(s, '.')[2])
        oct2 !== nothing && 16 <= oct2 <= 31 && return 3
    end
    startswith(s, "169.254.") && return 0   # link-local: last resort
    startswith(s, "100.")     && return 1   # CGNAT / Tailscale-style
    return 2                                  # other routable address
end

"""
    detect_lan_ip() -> Union{String, Nothing}

Best-effort detection of this machine's primary LAN IPv4 address. Enumerates all
interfaces and prefers ordinary private-LAN ranges over VPN/CGNAT virtual ones
(see [`_lan_ip_rank`](@ref)). Returns `nothing` if none found. Used to build a
`proxy_url` so Bonito advertises a remotely-reachable address instead of
`localhost`.
"""
function detect_lan_ip()
    try
        addrs = String[]
        for ip in Sockets.getipaddrs(Sockets.IPv4)
            s = string(ip)
            s == "127.0.0.1" || push!(addrs, s)
        end
        isempty(addrs) && return nothing
        return argmax(_lan_ip_rank, addrs)
    catch
        # Fall back to the single-address API on older Sockets.
        try
            ip = Sockets.getipaddr()
            ip isa Sockets.IPv4 || return nothing
            s = string(ip)
            return s == "127.0.0.1" ? nothing : s
        catch
            return nothing
        end
    end
end

"""
    resolve_proxy_url(proxy_url, host, port) -> String

Decide what external URL Bonito should advertise to clients for fetching its JS
runtime, assets, and websocket.

- If `proxy_url` is given explicitly, use it verbatim.
- Else, when binding to all interfaces (`0.0.0.0` / `::`), return `"."`: assets
  and the websocket resolve relative to whatever origin the browser loaded the
  page from, so LAN, VPN (Tailscale/OpenVPN), and localhost clients all work
  without guessing which interface IP the client reaches the server through.
- Else return `""` (Bonito's localhost-only default).
"""
function resolve_proxy_url(proxy_url::Union{Nothing, String}, host::String, port::Integer)
    proxy_url === nothing || return proxy_url
    host in ("0.0.0.0", "::") && return "."
    return ""
end

"""
    remote_url_hint(purl, port) -> String

Human-readable remote URL for startup banners. In origin-relative mode
(`purl == "."`) returns a best-effort `http://<lan-ip>:<port>` hint (any
address that reaches the server works, e.g. a VPN IP); otherwise returns
`purl` itself. Empty string if there is nothing useful to show.
"""
function remote_url_hint(purl::String, port::Integer)
    if purl == "."
        ip = detect_lan_ip()
        return ip === nothing ? "" : "http://$ip:$port"
    end
    return purl
end

# ============================================================================
# Export public API
# ============================================================================

export detect_lan_ip, resolve_proxy_url
export BRDFData, BRDFModel, BRDFOptimizer
export MeasuredBRDF
export lookup, lookup_batch, brdf_value
export vec3, normalize_vec, rgb, n_samples_str
export FORMAT_REGISTRY, MODEL_REGISTRY, OPTIMIZER_REGISTRY
export register_format!, register_model!, register_optimizer!
export evaluate, display_parameters, shader_parameters
export OptimizerOption, optimizer_options
