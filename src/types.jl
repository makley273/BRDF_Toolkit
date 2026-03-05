"""
Core type definitions for the BRDF Platform.

This file defines the abstract types and core data structures used throughout
the platform. All plugins must implement interfaces defined here.
"""

using StaticArrays
using Colors

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
- `evaluate(model, wi, wo)` - Compute BRDF value (Julia)
- `glsl_shader(model)` - Return GLSL shader code
- `shader_parameters(model)` - Extract parameters for uniforms
"""
abstract type BRDFModel <: BRDFData end

"""
    BRDFOptimizer

Abstract supertype for fitting algorithms.
Each optimizer must implement:
- `fit_brdf(optimizer, model_type, data)` - Fit model to data
"""
abstract type BRDFOptimizer end


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

# Convenience overload for tuples
function brdf_value(brdf::MeasuredBRDF, wi::Tuple{T,T,T}, wo::Tuple{T,T,T}) where T
    wi_vec = SVector{3, Float32}(wi[1], wi[2], wi[3])
    wo_vec = SVector{3, Float32}(wo[1], wo[2], wo[3])
    return brdf_value(brdf, wi_vec, wo_vec)
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
    evaluate(model::BRDFModel, wi::SVector{3,Float32}, wo::SVector{3,Float32}) -> RGB{Float32}

Compute the BRDF value for incident direction `wi` and outgoing direction `wo`.
Both vectors are normalized and point away from the surface.
The surface normal is assumed to be (0, 0, 1).

Must be implemented by every BRDFModel plugin.
"""
function evaluate end

"""
    glsl_shader(model::BRDFModel) -> Tuple{String, Vector{String}}

Return `(glsl_code, uniform_names)`.
- `glsl_code`: GLSL snippet defining `vec3 brdf_evaluate(vec3 wi, vec3 wo, vec3 N)`
  plus any `uniform` declarations it needs.
- `uniform_names`: list of uniform variable names declared in the snippet.

Must be implemented by every BRDFModel plugin.
"""
function glsl_shader end

"""
    shader_parameters(model::BRDFModel) -> Dict{String, Any}

Return a mapping from uniform name to its current value, matching the names
returned by `glsl_shader`.

Must be implemented by every BRDFModel plugin.
"""
function shader_parameters end

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
    @info "Registered optimizer plugin: $name"
    return nothing
end

# ============================================================================
# Export public API
# ============================================================================

export BRDFData, BRDFModel, BRDFOptimizer
export MeasuredBRDF
export lookup, lookup_batch, brdf_value
export vec3, normalize_vec, rgb, n_samples_str
export FORMAT_REGISTRY, MODEL_REGISTRY, OPTIMIZER_REGISTRY
export register_format!, register_model!, register_optimizer!
export evaluate, glsl_shader, shader_parameters
