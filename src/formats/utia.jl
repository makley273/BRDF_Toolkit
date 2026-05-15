"""
UTIA Anisotropic BRDF Format Loader

Loads BRDF data from the UTIA binary format (.bin files).

# Format Specification
- Fixed size: 1,990,656 bytes (3 planes × 6 × 48 × 6 × 48 × 8 bytes per Float64)
- Dimensions: nti=6 × npi=48 × ntv=6 × npv=48 (per color plane)
- Angle steps: stepT=15° (theta), stepP=7.5° (phi)
- Channels: RGB (3 stacked planes, plane-major)
- Anisotropic materials supported
- Data stored as doubles in raw binary (no header)

Adapted from the JS reference implementation in
`Proseminar/BRDF/src/unified/unified-brdf.js` (class UTIABrdf).

# Reference
Filip, J. and Vavra, R. "Template-Based Sampling of Anisotropic BRDFs",
Computer Graphics Forum (Pacific Graphics 2014).
"""

using StaticArrays
using LinearAlgebra
using Colors

# ============================================================================
# UTIA Format Type
# ============================================================================

"""
    UTIAFormat <: BRDFData

Format handler for UTIA binary BRDF files.
"""
struct UTIAFormat <: BRDFData end

# ============================================================================
# Constants
# ============================================================================

const UTIA_NTI = 6
const UTIA_NTV = 6
const UTIA_STEP_T = 15.0f0   # degrees
const UTIA_STEP_P = 7.5f0    # degrees
const UTIA_NPI = Int(360 ÷ UTIA_STEP_P)  # 48
const UTIA_NPV = UTIA_NPI                 # 48
const UTIA_PLANES = 3
const UTIA_STRIDE_PLANES = UTIA_NTI * UTIA_NPI * UTIA_NTV * UTIA_NPV  # 82_944
const UTIA_TOTAL_FLOATS = UTIA_PLANES * UTIA_STRIDE_PLANES             # 248_832
const UTIA_EXPECTED_SIZE = UTIA_TOTAL_FLOATS * sizeof(Float64)         # 1_990_656

# UTIA approximates the same RGB wavelengths as MERL for visualization purposes
const UTIA_WAVELENGTHS = Float32[630, 532, 465]

# ============================================================================
# Format Detection
# ============================================================================

"""
    detect_format(::Type{UTIAFormat}, filepath::String) -> Float64

Detect if a file is in UTIA binary format.
Returns confidence score 0.0-1.0.
"""
function detect_format(::Type{UTIAFormat}, filepath::String)::Float64
    if !isfile(filepath)
        return 0.0
    end

    confidence = 0.0
    if endswith(lowercase(filepath), ".bin")
        confidence += 0.3
    end
    if filesize(filepath) == UTIA_EXPECTED_SIZE
        confidence += 0.7
    end
    return confidence
end

# ============================================================================
# UTIA Internal Data Structure
# ============================================================================

"""
    UTIAData

Internal storage for UTIA BRDF data.

# Storage Layout
Plane-major flat array of length `3 * UTIA_STRIDE_PLANES`. For color plane `p`,
sample at grid coordinates `(ti, pi, tv, pv)` (all 0-based) is stored at:

    p * UTIA_STRIDE_PLANES + (UTIA_NPV * UTIA_NTV) * (UTIA_NPI * ti + pi)
                           + UTIA_NPV * tv + pv
"""
struct UTIAData
    samples::Vector{Float32}   # length 3 * UTIA_STRIDE_PLANES
    nti::Int
    ntv::Int
    npi::Int
    npv::Int
    stepT::Float32             # degrees
    stepP::Float32             # degrees
end

n_samples_str(d::UTIAData) = string(d.nti * d.ntv * d.npi * d.npv)

# ============================================================================
# Lookup Implementation
# ============================================================================

@inline function _utia_flat_index(d::UTIAData, ti::Int, pi_idx::Int,
                                  tv::Int, pv::Int, p::Int)::Int
    # 0-based indices in, 1-based Julia index out.
    nv = d.npv * d.ntv
    return p * (d.nti * d.npi * d.ntv * d.npv) +
           nv * (d.npi * ti + pi_idx) +
           d.npv * tv + pv + 1
end

@inline _utia_wrap(i::Int, n::Int) = ((i % n) + n) % n
@inline _utia_clamp(i::Int, n::Int) = max(0, min(i, n - 1))

"""
    _brdf_value(data::UTIAData, wavelengths, wi, wo) -> RGB{Float32}

Quadrilinear interpolation in 4D angle space (θᵢ, φᵢ, θᵥ, φᵥ).

Returns zero outside the upper hemisphere. Raw measured values are returned
without normalization (the JS reference divides by `MAXIMUM_TOTAL ≈ 38.336`;
we keep raw values, mirroring how the MERL loader exposes pre-scale samples).
"""
function _brdf_value(data::UTIAData,
                     wavelengths::Vector{Float32},
                     wi::SVector{3, Float32},
                     wo::SVector{3, Float32})::RGB{Float32}

    # Convert to spherical angles
    θi = acos(clamp(wi[3], -1.0f0, 1.0f0))
    θv = acos(clamp(wo[3], -1.0f0, 1.0f0))
    if θi > Float32(π/2) || θv > Float32(π/2)
        return RGB{Float32}(0, 0, 0)
    end

    two_pi = Float32(2π)
    φi = mod(atan(wi[2], wi[1]), two_pi)
    φv = mod(atan(wo[2], wo[1]), two_pi)

    r2d = Float32(180 / π)
    inv_step_t = 1.0f0 / data.stepT
    inv_step_p = 1.0f0 / data.stepP

    # ---- θᵢ ----
    ti_val = min(θi * r2d, data.stepT * (data.nti - 1))
    ti0 = min(Int(floor(ti_val * inv_step_t)), data.nti - 2)
    ti1 = (ti0 + 1) % data.nti
    wti1 = ti_val - data.stepT * ti0
    wti0 = data.stepT * ti1 - ti_val
    wti_sum = wti0 + wti1
    wti0 /= wti_sum;  wti1 /= wti_sum

    # ---- φᵢ ----
    pi_val = mod(φi * r2d, 360.0f0)
    pi0 = Int(floor(pi_val * inv_step_p))
    pi1 = (pi0 + 1) % data.npi
    wpi1 = pi_val - data.stepP * pi0
    wpi0 = data.stepP * pi1 - pi_val
    wpi_sum = wpi0 + wpi1
    wpi0 /= wpi_sum;  wpi1 /= wpi_sum

    # ---- θᵥ ----
    tv_val = min(θv * r2d, data.stepT * (data.ntv - 1))
    tv0 = min(Int(floor(tv_val * inv_step_t)), data.ntv - 2)
    tv1 = (tv0 + 1) % data.ntv
    wtv1 = tv_val - data.stepT * tv0
    wtv0 = data.stepT * tv1 - tv_val
    wtv_sum = wtv0 + wtv1
    wtv0 /= wtv_sum;  wtv1 /= wtv_sum

    # ---- φᵥ (JS uses linear weights here, no normalization) ----
    pv_val = mod(φv * r2d, 360.0f0)
    pv0 = Int(floor(pv_val * inv_step_p))
    pv1 = (pv0 + 1) % data.npv
    wpv1 = pv_val * inv_step_p - pv0
    wpv0 = 1.0f0 - wpv1

    # Wrap φ indices defensively (should already be in range)
    pi0 = _utia_wrap(pi0, data.npi);  pi1 = _utia_wrap(pi1, data.npi)
    pv0 = _utia_wrap(pv0, data.npv);  pv1 = _utia_wrap(pv1, data.npv)
    ti0 = _utia_clamp(ti0, data.nti); ti1 = _utia_clamp(ti1, data.nti)
    tv0 = _utia_clamp(tv0, data.ntv); tv1 = _utia_clamp(tv1, data.ntv)

    # Quadrilinear interpolation per color plane
    out = MVector{3, Float32}(0, 0, 0)
    @inbounds for p in 0:(UTIA_PLANES - 1)
        s = data.samples
        s0000 = s[_utia_flat_index(data, ti0, pi0, tv0, pv0, p)]
        s0001 = s[_utia_flat_index(data, ti0, pi0, tv0, pv1, p)]
        s0010 = s[_utia_flat_index(data, ti0, pi0, tv1, pv0, p)]
        s0011 = s[_utia_flat_index(data, ti0, pi0, tv1, pv1, p)]
        s0100 = s[_utia_flat_index(data, ti0, pi1, tv0, pv0, p)]
        s0101 = s[_utia_flat_index(data, ti0, pi1, tv0, pv1, p)]
        s0110 = s[_utia_flat_index(data, ti0, pi1, tv1, pv0, p)]
        s0111 = s[_utia_flat_index(data, ti0, pi1, tv1, pv1, p)]
        s1000 = s[_utia_flat_index(data, ti1, pi0, tv0, pv0, p)]
        s1001 = s[_utia_flat_index(data, ti1, pi0, tv0, pv1, p)]
        s1010 = s[_utia_flat_index(data, ti1, pi0, tv1, pv0, p)]
        s1011 = s[_utia_flat_index(data, ti1, pi0, tv1, pv1, p)]
        s1100 = s[_utia_flat_index(data, ti1, pi1, tv0, pv0, p)]
        s1101 = s[_utia_flat_index(data, ti1, pi1, tv0, pv1, p)]
        s1110 = s[_utia_flat_index(data, ti1, pi1, tv1, pv0, p)]
        s1111 = s[_utia_flat_index(data, ti1, pi1, tv1, pv1, p)]

        # Collapse pv → tv → pi → ti (same order as JS reference)
        c000 = s0000 * wpv0 + s0001 * wpv1
        c001 = s0010 * wpv0 + s0011 * wpv1
        c010 = s0100 * wpv0 + s0101 * wpv1
        c011 = s0110 * wpv0 + s0111 * wpv1
        c100 = s1000 * wpv0 + s1001 * wpv1
        c101 = s1010 * wpv0 + s1011 * wpv1
        c110 = s1100 * wpv0 + s1101 * wpv1
        c111 = s1110 * wpv0 + s1111 * wpv1

        c00 = c000 * wtv0 + c001 * wtv1
        c01 = c010 * wtv0 + c011 * wtv1
        c10 = c100 * wtv0 + c101 * wtv1
        c11 = c110 * wtv0 + c111 * wtv1

        c0 = c00 * wpi0 + c01 * wpi1
        c1 = c10 * wpi0 + c11 * wpi1

        out[p + 1] = c0 * wti0 + c1 * wti1
    end

    return RGB{Float32}(out[1], out[2], out[3])
end

# ============================================================================
# Loader
# ============================================================================

"""
    load_brdf(::Type{UTIAFormat}, filepath::String) -> MeasuredBRDF

Load BRDF data from a UTIA binary file.
"""
function load_brdf(::Type{UTIAFormat}, filepath::String)::MeasuredBRDF
    if !isfile(filepath)
        empty_data = UTIAData(zeros(Float32, 1), UTIA_NTI, UTIA_NTV,
                              UTIA_NPI, UTIA_NPV, UTIA_STEP_T, UTIA_STEP_P)
        return MeasuredBRDF(
            empty_data, UTIA_WAVELENGTHS, basename(filepath);
            source_format = :utia, status = :failed,
            error_msg = "File not found: $filepath"
        )
    end

    fsize = filesize(filepath)
    if fsize != UTIA_EXPECTED_SIZE
        @warn "UTIA file size mismatch: expected $UTIA_EXPECTED_SIZE, got $fsize"
    end

    try
        raw = open(filepath, "r") do io
            buf = Vector{Float64}(undef, UTIA_TOTAL_FLOATS)
            read!(io, buf)
            buf
        end
        samples = Float32.(raw)

        utia_data = UTIAData(
            samples, UTIA_NTI, UTIA_NTV, UTIA_NPI, UTIA_NPV,
            UTIA_STEP_T, UTIA_STEP_P
        )

        return MeasuredBRDF(
            utia_data, UTIA_WAVELENGTHS,
            replace(basename(filepath), ".bin" => "");
            source_format = :utia, status = :ok, error_msg = nothing
        )
    catch e
        @error "Failed to load UTIA file" exception=(e, catch_backtrace())
        empty_data = UTIAData(zeros(Float32, 1), UTIA_NTI, UTIA_NTV,
                              UTIA_NPI, UTIA_NPV, UTIA_STEP_T, UTIA_STEP_P)
        return MeasuredBRDF(
            empty_data, UTIA_WAVELENGTHS, basename(filepath);
            source_format = :utia, status = :failed,
            error_msg = "Error reading file: $(sprint(showerror, e))"
        )
    end
end

# ============================================================================
# Register Plugin
# ============================================================================

register_format!(UTIAFormat)
