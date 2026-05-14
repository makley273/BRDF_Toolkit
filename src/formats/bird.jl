"""
BiRD JSON BRDF Format Loader

Loads BRDF data from the BiRD (.bird / .json) format used by the BiRD-project
at Aalto University.

# Format Specification
- JSON-based, irregularly sampled BRDF data.
- Root object has exactly two keys: "metadata" and "data".
- `metadata.type` must equal "BRDF".
- `data` contains five required variable objects: theta_i, phi_i, theta_r,
  phi_r, BRDF. All `values` arrays have identical length N.
- Optional variables: wavelength_i, wavelength_r, polarization_i,
  polarization_r, uBRDF, adhoc_variables.

This basic loader reads the optional fields (wavelength_i, polarization_i,
polarization_r, adhoc_variables) but does not interpret them — they are
stored raw on the data struct for future use. `wavelength_r` (the observed /
reflected wavelength) is parsed and used to populate
`MeasuredBRDF.wavelengths`.

# Reference
- BiRD JSON Schema v1.0
  https://github.com/BiRD-project/BiRD_view
- Reference parser: BiRD_view/parse_brdf_json.py
"""

using StaticArrays
using LinearAlgebra
using JSON3

# Types are already in scope (included in BRDFPlatform module).

# ============================================================================
# BiRD Format Type
# ============================================================================

"""
    BiRDFormat <: BRDFData

Format handler for BiRD JSON BRDF files.
"""
struct BiRDFormat <: BRDFData end

# ============================================================================
# Constants
# ============================================================================

const BIRD_DEFAULT_WAVELENGTH = Float32[550.0]
const BIRD_IDW_K = 4              # number of nearest neighbours for IDW lookup
const BIRD_IDW_EPS = 1.0f-8       # exact-hit threshold (squared distance)

# ============================================================================
# Internal Data Structure
# ============================================================================

"""
    BiRDData

Internal storage for a parsed BiRD BRDF file.

Required angle arrays are stored in **radians** (converted from "deg"/"°" if
necessary). BRDF values are in sr⁻¹.
"""
struct BiRDData
    theta_i::Vector{Float32}
    phi_i::Vector{Float32}
    theta_r::Vector{Float32}
    phi_r::Vector{Float32}
    brdf::Vector{Float32}

    wavelength_r::Union{Nothing, Vector{Float32}}   # nm
    wavelength_i_raw::Any
    polarization_i_raw::Any
    polarization_r_raw::Any
    adhoc_raw::Any
    metadata::Dict{String, Any}
end

n_samples_str(d::BiRDData) = string(length(d.brdf))

# ============================================================================
# Helpers
# ============================================================================

"""
    _angle_to_radians(unit, values) -> Vector{Float32}

Convert an array of angle values to radians, given the BiRD-spec unit string
("rad", "deg", "°"). "deg" and "°" are equivalent (spec §3.4).
"""
function _angle_to_radians(unit::AbstractString, values)::Vector{Float32}
    vs = Float32.(values)
    if unit == "rad"
        return vs
    elseif unit == "deg" || unit == "°"
        return Float32.(vs .* (Float32(π) / 180.0f0))
    else
        error("Unknown angle unit: \"$unit\" (expected \"rad\", \"deg\", or \"°\")")
    end
end

"""
    _wavelength_to_nm(unit, values) -> Vector{Float32}

Convert wavelength array to nm. Recognised units: "nm", "μm" (U+03BC).
"""
function _wavelength_to_nm(unit::AbstractString, values)::Vector{Float32}
    vs = Float32.(values)
    if unit == "nm"
        return vs
    elseif unit == "μm" || unit == "um"
        return Float32.(vs .* 1000.0f0)
    else
        error("Unknown wavelength unit: \"$unit\"")
    end
end

# Convert JSON3 nested structures into plain Julia Dict{String,Any} so that
# downstream code (and the failure-path constructor) does not depend on JSON3
# types.
_to_plain(x::JSON3.Object) = Dict{String,Any}(string(k) => _to_plain(v) for (k, v) in x)
_to_plain(x::JSON3.Array)  = Any[_to_plain(v) for v in x]
_to_plain(x) = x

"""
    _empty_bird_data() -> BiRDData

Return an empty BiRDData used for failure-path MeasuredBRDF construction.
"""
function _empty_bird_data()
    BiRDData(
        Float32[], Float32[], Float32[], Float32[], Float32[],
        nothing, nothing, nothing, nothing, nothing,
        Dict{String,Any}(),
    )
end

# ============================================================================
# Format Detection
# ============================================================================

"""
    detect_format(::Type{BiRDFormat}, filepath::String) -> Float64

Detect whether a file is a BiRD JSON BRDF. Confidence score in [0, 1].
Never throws — JSON parse failures return 0.0.
"""
function detect_format(::Type{BiRDFormat}, filepath::String)::Float64
    if !isfile(filepath)
        return 0.0
    end

    confidence = 0.0
    lower = lowercase(filepath)
    if endswith(lower, ".brdf") || endswith(lower, ".bird")
        confidence += 0.4
    elseif endswith(lower, ".json")
        confidence += 0.2
    end

    try
        text = read(filepath, String)
        obj  = JSON3.read(text)
        if haskey(obj, :metadata) && haskey(obj, :data)
            md = obj.metadata
            if haskey(md, :type) && md.type == "BRDF"
                confidence += 0.6
            end
        end
    catch
        return 0.0
    end

    return min(confidence, 1.0)
end

# ============================================================================
# Lookup (IDW over k-nearest samples in 4D angle space)
# ============================================================================

@inline function _angular_diff(a::Float32, b::Float32)::Float32
    d = abs(a - b)
    twoπ = Float32(2π)
    return min(d, twoπ - d)
end

"""
    _brdf_value(data::BiRDData, wavelengths, wi, wo) -> RGB{Float32}

Look up the BRDF for the given incident/outgoing directions using
inverse-distance weighting over the `BIRD_IDW_K` nearest samples in 4D
angle space. Returns RGB(b,b,b) — the basic loader does not perform a
spectral→RGB mapping.
"""
function _brdf_value(data::BiRDData,
                     wavelengths::Vector{Float32},
                     wi::SVector{3, Float32},
                     wo::SVector{3, Float32})::RGB{Float32}

    n = length(data.brdf)
    if n == 0
        return RGB{Float32}(0, 0, 0)
    end

    # Convert directions → spherical angles (radians) in surface frame.
    θi = acos(clamp(wi[3], -1.0f0, 1.0f0))
    φi = mod(atan(wi[2], wi[1]), Float32(2π))
    θr = acos(clamp(wo[3], -1.0f0, 1.0f0))
    φr = mod(atan(wo[2], wo[1]), Float32(2π))

    # Compute squared angle-space distances to all samples.
    d2 = Vector{Float32}(undef, n)
    @inbounds for s in 1:n
        dθi = θi - data.theta_i[s]
        dθr = θr - data.theta_r[s]
        dφi = _angular_diff(φi, data.phi_i[s])
        dφr = _angular_diff(φr, data.phi_r[s])
        d2[s] = dθi*dθi + dθr*dθr + dφi*dφi + dφr*dφr
    end

    # Exact-hit short-circuit (avoids division by zero).
    @inbounds for s in 1:n
        if d2[s] < BIRD_IDW_EPS
            v = data.brdf[s]
            v = max(0.0f0, v)
            return RGB{Float32}(v, v, v)
        end
    end

    # Pick k nearest indices (small k → linear scans are fine).
    k = min(BIRD_IDW_K, n)
    nearest = partialsortperm(d2, 1:k)

    wsum = 0.0f0
    bsum = 0.0f0
    @inbounds for s in nearest
        w = 1.0f0 / (d2[s] + BIRD_IDW_EPS)
        wsum += w
        bsum += w * data.brdf[s]
    end

    b = max(0.0f0, bsum / wsum)
    return RGB{Float32}(b, b, b)
end

# ============================================================================
# Loading
# ============================================================================

"""
    load_brdf(::Type{BiRDFormat}, filepath::String) -> MeasuredBRDF

Load and parse a BiRD JSON BRDF file. Returns a `MeasuredBRDF` wrapping a
`BiRDData`. On any error returns a MeasuredBRDF with `status = :failed` and
a populated `error_msg`.
"""
function load_brdf(::Type{BiRDFormat}, filepath::String)::MeasuredBRDF
    fname = basename(filepath)

    if !isfile(filepath)
        return MeasuredBRDF(
            _empty_bird_data(), BIRD_DEFAULT_WAVELENGTH, fname;
            source_format = :bird, status = :failed,
            error_msg = "File not found: $filepath",
        )
    end

    try
        text = read(filepath, String)
        obj  = JSON3.read(text)

        haskey(obj, :metadata) || error("Missing top-level \"metadata\" key")
        haskey(obj, :data)     || error("Missing top-level \"data\" key")

        md = obj.metadata
        if !haskey(md, :type) || md.type != "BRDF"
            error("metadata.type is not \"BRDF\"")
        end

        d = obj.data
        for req in (:theta_i, :phi_i, :theta_r, :phi_r, :BRDF)
            haskey(d, req) || error("Missing required data variable: \"$req\"")
        end

        θi = _angle_to_radians(string(d.theta_i.unit), d.theta_i.values)
        φi = _angle_to_radians(string(d.phi_i.unit),   d.phi_i.values)
        θr = _angle_to_radians(string(d.theta_r.unit), d.theta_r.values)
        φr = _angle_to_radians(string(d.phi_r.unit),   d.phi_r.values)

        brdf_unit = string(d.BRDF.unit)
        brdf_unit in ("sr^-1", "1/sr") ||
            error("BRDF unit must be \"sr^-1\" or \"1/sr\", got \"$brdf_unit\"")
        brdf_vals = Float32.(collect(d.BRDF.values))

        N = length(brdf_vals)
        for (name, arr) in (("theta_i", θi), ("phi_i", φi),
                            ("theta_r", θr), ("phi_r", φr))
            length(arr) == N ||
                error("Length mismatch: $name has $(length(arr)) entries, expected $N")
        end

        # Optional: wavelength_r — parsed for the wavelengths field on MeasuredBRDF.
        wavelength_r = nothing
        wavelengths  = BIRD_DEFAULT_WAVELENGTH
        if haskey(d, :wavelength_r)
            wr_vals = _wavelength_to_nm(string(d.wavelength_r.unit),
                                        d.wavelength_r.values)
            length(wr_vals) == N ||
                error("Length mismatch: wavelength_r has $(length(wr_vals)) entries, expected $N")
            wavelength_r = wr_vals
            unique_wl = sort(unique(wr_vals))
            if !isempty(unique_wl)
                wavelengths = Float32.(unique_wl)
            end
        end

        # Read-but-don't-interpret optional fields.
        wavelength_i_raw   = haskey(d, :wavelength_i)   ? _to_plain(d.wavelength_i)   : nothing
        polarization_i_raw = haskey(d, :polarization_i) ? _to_plain(d.polarization_i) : nothing
        polarization_r_raw = haskey(d, :polarization_r) ? _to_plain(d.polarization_r) : nothing
        adhoc_raw          = haskey(d, :adhoc_variables) ? _to_plain(d.adhoc_variables) : nothing

        metadata = _to_plain(md)

        bird_data = BiRDData(
            θi, φi, θr, φr, brdf_vals,
            wavelength_r,
            wavelength_i_raw,
            polarization_i_raw,
            polarization_r_raw,
            adhoc_raw,
            metadata,
        )

        # Derive a human-readable name.
        name = if haskey(md, :id) && md.id isa AbstractString
            last_seg = split(string(md.id), "/")[end]
            isempty(last_seg) ? fname : String(last_seg)
        else
            String(replace(fname, r"\.(brdf|bird|json)$"i => ""))
        end

        return MeasuredBRDF(
            bird_data, wavelengths, name;
            source_format = :bird, status = :ok, error_msg = nothing,
        )

    catch e
        @error "Failed to load BiRD file" exception=(e, catch_backtrace())
        return MeasuredBRDF(
            _empty_bird_data(), BIRD_DEFAULT_WAVELENGTH, fname;
            source_format = :bird, status = :failed,
            error_msg = "Error reading file: $(sprint(showerror, e))",
        )
    end
end

# ============================================================================
# Register Plugin
# ============================================================================

register_format!(BiRDFormat)
