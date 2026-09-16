"""
MERL Binary BRDF Format Loader

Loads BRDF data from the MERL binary format (.binary files).

# Format Specification
- Fixed size: 17,496,000 bytes (90 × 90 × 180 × 3 channels × 8 bytes per double)
- Dimensions: 90 theta_h × 90 theta_d × 180 phi_d
- Parameterization: Rusinkiewicz half-angle
- Channels: RGB (approximated wavelengths: 630nm, 532nm, 465nm)
- Isotropic materials only (phi_h = 0)
- Data stored as doubles in binary format

# Reference
Wojciech Matusik, Hanspeter Pfister, Matt Brand and Leonard McMillan.
"A Data-Driven Reflectance Model". ACM Transactions on Graphics, 2003.
"""

using StaticArrays
using LinearAlgebra

# Types are already in scope since this file is included in BRDFPlatform module
# No imports needed - BRDFData, MeasuredBRDF, etc. are already available

# ============================================================================
# MERL Format Type
# ============================================================================

"""
    MERLFormat <: BRDFData

Format handler for MERL binary BRDF files.
"""
struct MERLFormat <: BRDFData end

# ============================================================================
# Constants
# ============================================================================

const MERL_SAMPLING_RES_THETA_H = 90
const MERL_SAMPLING_RES_THETA_D = 90
const MERL_SAMPLING_RES_PHI_D = 180

# MERL files have a 12-byte header with dimensions (3 Int32 values)
const MERL_HEADER_SIZE = 12

# Expected file size: 12-byte header + 90 * 90 * 180 * 3 channels * 8 bytes (Float64)
const MERL_EXPECTED_SIZE = 34992012

# MERL approximates these wavelengths for RGB
const MERL_WAVELENGTHS = Float32[630, 532, 465]  # Red, Green, Blue (nm)

# MERL stores values with different scale factors per channel
# Reference: merl.cpp from MERL BRDF Database
const MERL_RED_SCALE = 1.0 / 1500.0
const MERL_GREEN_SCALE = 1.15 / 1500.0
const MERL_BLUE_SCALE = 1.66 / 1500.0

# ============================================================================
# Format Detection
# ============================================================================

"""
    detect_format(::Type{MERLFormat}, filepath::String) -> Float64

Detect if a file is in MERL binary format.
Returns confidence score 0.0-1.0.
"""
function detect_format(::Type{MERLFormat}, filepath::String)::Float64
    confidence = 0.0

    # Check file exists
    if !isfile(filepath)
        return 0.0
    end

    # Check extension
    if endswith(lowercase(filepath), ".binary")
        confidence += 0.2
    end

    # Check file size
    file_size = filesize(filepath)
    if file_size == MERL_EXPECTED_SIZE
        confidence += 0.8
    end

    return confidence
end

# ============================================================================
# MERL Internal Data Structure
# ============================================================================

"""
    MERLData

Internal storage for MERL BRDF data.

# Storage Layout (Matches C++ reference)
The file is stored as 3 blocks of data (Red, Green, Blue).
Within each color block, the loop order is:
1. Theta_H (Slowest, outer loop)
2. Theta_D
3. Phi_D   (Fastest, inner loop)

Index = pd + td * n_phi_d + th * n_phi_d * n_theta_d
"""
struct MERLData
    samples::Matrix{Float32}  # 3 × n_samples
    n_theta_h::Int
    n_theta_d::Int
    n_phi_d::Int
end

n_samples_str(d::MERLData) = string(d.n_theta_h * d.n_theta_d * d.n_phi_d)

# ============================================================================
# Rusinkiewicz Parameterization Utilities (Continuous)
# ============================================================================

"""
    map_theta_h_continuous(theta_h_rad, n_res) -> Float32

Returns 0-based floating point index for theta_h using sqrt mapping.
"""
function map_theta_h_continuous(theta_h_rad::Float32, n_res::Int)::Float32
    # Mapping: Index proportional to sqrt(angle)
    # C++ logic: sqrt(theta_deg / 90.0 * n_res) * n_res => sqrt(theta_rad / (pi/2)) * n_res
    
    val = sqrt(max(0.0f0, theta_h_rad) / (Float32(π) / 2.0f0)) * Float32(n_res)
    
    # Clamp to valid range [0, N-1]
    # We use N-1.0001 to ensure the 'ceil' index doesn't overflow later
    return clamp(val, 0.0f0, Float32(n_res) - 1.0001f0)
end

"""
    map_theta_d_continuous(theta_d_rad, n_res) -> Float32

Returns 0-based floating point index for theta_d (Linear).
"""
function map_theta_d_continuous(theta_d_rad::Float32, n_res::Int)::Float32
    # C++ uses linear mapping for theta_d (unlike previous incorrect version)
    val = (theta_d_rad / (Float32(π) / 2.0f0)) * Float32(n_res)
    return clamp(val, 0.0f0, Float32(n_res) - 1.0001f0)
end

"""
    map_phi_d_continuous(phi_d_rad, n_res) -> Float32

Returns 0-based floating point index for phi_d.
Handles reciprocity (BRDF(phi) = BRDF(phi + pi)).
"""
function map_phi_d_continuous(phi_d_rad::Float32, n_res::Int)::Float32
    # Wrap phi if negative (Isotropic reciprocity)
    curr_phi = phi_d_rad
    if curr_phi < 0.0f0
        curr_phi += Float32(π)
    end
    
    val = (curr_phi / Float32(π)) * Float32(n_res)
    return clamp(val, 0.0f0, Float32(n_res) - 1.0001f0)
end

# ============================================================================
# Lookup Implementation (Trilinear)
# ============================================================================

"""
    _brdf_value(data::MERLData, wavelengths, wi, wo) -> RGB{Float32}

Lookup BRDF value for MERL format using Rusinkiewicz parameterization and Trilinear Interpolation.
"""
function _brdf_value(data::MERLData,
                wavelengths::Vector{Float32},
                wi::SVector{3, Float32},
                wo::SVector{3, Float32})::RGB{Float32}

    # --- 1. Geometry Calculation ---
    wi_len = norm(wi)
    wo_len = norm(wo)
    eps = Float32(1e-6)

    if wi_len < eps || wo_len < eps
        return RGB{Float32}(0, 0, 0)
    end

    wi_n = wi / wi_len
    wo_n = wo / wo_len

    h = wi_n + wo_n
    h_len = norm(h)
    
    if h_len < eps
        return RGB{Float32}(0, 0, 0)
    end

    h_n = h / h_len
    
    # theta_h: angle between H and N (0,0,1)
    theta_h = acos(clamp(h_n[3], -1.0f0, 1.0f0))

    # theta_d: angle between H and wi
    cos_theta_d = dot(wi_n, h_n)
    theta_d = acos(clamp(cos_theta_d, -1.0f0, 1.0f0))

    # phi_d: Rusinkiewicz difference azimuthal angle
    # Build an orthonormal frame where h_n is the z-axis, then express wi in that frame.
    # The azimuthal angle of wi in this frame is phi_d.
    #
    # Frame construction (Duff et al. 2017 building-an-orthonormal-basis):
    #   t = tangent aligned with projection of N onto the plane ⊥ h
    #   b = h × t
    # Then phi_d = atan2(dot(wi, b), dot(wi, t))

    N = SVector{3, Float32}(0.0f0, 0.0f0, 1.0f0)

    # Project N onto the plane perpendicular to h_n
    t = N - dot(N, h_n) * h_n
    t_len = norm(t)

    if t_len < eps
        # h is parallel to N (theta_h ≈ 0), phi_d is degenerate — use fallback frame
        t = SVector{3, Float32}(1.0f0, 0.0f0, 0.0f0)
        t = t - dot(t, h_n) * h_n
        t_len = norm(t)
        if t_len < eps
            t = SVector{3, Float32}(0.0f0, 1.0f0, 0.0f0)
            t = t - dot(t, h_n) * h_n
            t_len = norm(t)
        end
    end
    t = t / t_len

    b = cross(h_n, t)

    # Express wi in the (t, b, h_n) frame and extract azimuthal angle
    phi_d = atan(dot(wi_n, b), dot(wi_n, t))

    # --- 2. Continuous Index Mapping ---
    # Get 0-based floating point coordinates (u, v, w)
    u_th = map_theta_h_continuous(theta_h, data.n_theta_h)
    v_td = map_theta_d_continuous(theta_d, data.n_theta_d)
    w_pd = map_phi_d_continuous(phi_d, data.n_phi_d)

    # --- 3. Trilinear Interpolation Setup ---
    
    # Calculate integer base indices (floor)
    idx_th = floor(Int, u_th)
    idx_td = floor(Int, v_td)
    idx_pd = floor(Int, w_pd)

    # Calculate weights (fractional part)
    frac_th = u_th - idx_th
    frac_td = v_td - idx_td
    frac_pd = w_pd - idx_pd

    # Strides (C++ Memory Layout: Phi Fastest -> ThetaD -> ThetaH Slowest)
    stride_pd = 1
    stride_td = data.n_phi_d
    stride_th = data.n_phi_d * data.n_theta_d
    
    # Calculate base linear index (0-based relative to start of channel block)
    # Note: idx_th, idx_td, idx_pd are 0-based
    base_idx = (idx_pd * stride_pd) + 
               (idx_td * stride_td) + 
               (idx_th * stride_th) + 
               1 # +1 for Julia 1-based indexing

    # Delta offsets for neighbors
    d_pd = stride_pd
    d_td = stride_td
    d_th = stride_th

    # --- 4. Interpolate ---
    
    # Pre-calculate blending weights
    w000 = (1-frac_th)*(1-frac_td)*(1-frac_pd)
    w001 = (1-frac_th)*(1-frac_td)*(frac_pd)
    w010 = (1-frac_th)*(frac_td)*(1-frac_pd)
    w011 = (1-frac_th)*(frac_td)*(frac_pd)
    w100 = (frac_th)*(1-frac_td)*(1-frac_pd)
    w101 = (frac_th)*(1-frac_td)*(frac_pd)
    w110 = (frac_th)*(frac_td)*(1-frac_pd)
    w111 = (frac_th)*(frac_td)*(frac_pd)

    # Helper to accumulate weighted sum for a channel row
    function accum_channel(row_idx)
        # Neighbor indices
        i000 = base_idx
        i001 = base_idx + d_pd
        i010 = base_idx + d_td
        i011 = base_idx + d_td + d_pd
        i100 = base_idx + d_th
        i101 = base_idx + d_th + d_pd
        i110 = base_idx + d_th + d_td
        i111 = base_idx + d_th + d_td + d_pd

        return data.samples[row_idx, i000] * w000 +
               data.samples[row_idx, i001] * w001 +
               data.samples[row_idx, i010] * w010 +
               data.samples[row_idx, i011] * w011 +
               data.samples[row_idx, i100] * w100 +
               data.samples[row_idx, i101] * w101 +
               data.samples[row_idx, i110] * w110 +
               data.samples[row_idx, i111] * w111
    end

    r = accum_channel(1)
    g = accum_channel(2)
    b = accum_channel(3)

    return RGB{Float32}(max(0.0f0, r), max(0.0f0, g), max(0.0f0, b))
end

# ============================================================================
# Data Loading
# ============================================================================

"""
    load_brdf(::Type{MERLFormat}, filepath::String) -> MeasuredBRDF

Load BRDF data from a MERL binary file.
"""
function load_brdf(::Type{MERLFormat}, filepath::String)::MeasuredBRDF
    # Validate file
    if !isfile(filepath)
        empty_data = MERLData(zeros(Float32, 3, 1), 1, 1, 1)
        return MeasuredBRDF(
            empty_data,
            MERL_WAVELENGTHS,
            basename(filepath),
            source_format = :merl,
            status = :failed,
            error_msg = "File not found: $filepath"
        )
    end

    file_size = filesize(filepath)
    if file_size != MERL_EXPECTED_SIZE
        @warn "File size mismatch: expected $MERL_EXPECTED_SIZE, got $file_size"
    end

    # Read binary data
    try
        data_block = open(filepath, "r") do io
            # Skip 12-byte header
            skip(io, MERL_HEADER_SIZE)

            n_samples = MERL_SAMPLING_RES_THETA_H *
                        MERL_SAMPLING_RES_THETA_D *
                        MERL_SAMPLING_RES_PHI_D

            raw_data = zeros(Float64, n_samples * 3)
            read!(io, raw_data)
            raw_data
        end

        n_samples = MERL_SAMPLING_RES_THETA_H *
                    MERL_SAMPLING_RES_THETA_D *
                    MERL_SAMPLING_RES_PHI_D

        # Extract RGB channels (stored as contiguous blocks)
        # [All Reds] [All Greens] [All Blues]
        red_channel   = Float32.(data_block[1:n_samples])
        green_channel = Float32.(data_block[n_samples+1:2*n_samples])
        blue_channel  = Float32.(data_block[2*n_samples+1:3*n_samples])

        # Apply MERL channel-specific scaling factors
        red_channel   .*= Float32(MERL_RED_SCALE)
        green_channel .*= Float32(MERL_GREEN_SCALE)
        blue_channel  .*= Float32(MERL_BLUE_SCALE)

        # Create internal data structure [3 wavelengths × n_samples]
        samples_matrix = zeros(Float32, 3, n_samples)
        samples_matrix[1, :] = red_channel
        samples_matrix[2, :] = green_channel
        samples_matrix[3, :] = blue_channel

        # Wrap in MERLData
        merl_data = MERLData(
            samples_matrix,
            MERL_SAMPLING_RES_THETA_H,
            MERL_SAMPLING_RES_THETA_D,
            MERL_SAMPLING_RES_PHI_D
        )

        return MeasuredBRDF(
            merl_data,
            MERL_WAVELENGTHS,
            replace(basename(filepath), ".binary" => ""),
            source_format = :merl,
            status = :ok,
            error_msg = nothing
        )

    catch e
        @error "Failed to load MERL file" exception=(e, catch_backtrace())
        empty_data = MERLData(zeros(Float32, 3, 1), 1, 1, 1)
        return MeasuredBRDF(
            empty_data,
            MERL_WAVELENGTHS,
            basename(filepath),
            source_format = :merl,
            status = :failed,
            error_msg = "Error reading file: $(sprint(showerror, e))"
        )
    end
end

# ============================================================================
# Register Plugin
# ============================================================================

register_format!(MERLFormat)