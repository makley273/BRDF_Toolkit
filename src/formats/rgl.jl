"""
RGL EPFL Material Database Loader (.bsdf tensor files)

Loads measured BRDFs from the RGL (Realistic Graphics Lab, EPFL) material
database, which stores data in the adaptive parameterization of

> Dupuy, J. and Jakob, W. "An Adaptive Parameterization for Efficient
> Material Acquisition and Rendering", SIGGRAPH Asia 2018.

This is a pure-Julia port of the eval path of the official reference loader
(`powitacq_rgb.h` / `powitacq_rgb.inl`, https://github.com/rgl-epfl/brdf-loader).
Only the RGB tensors are used (the spectral data present in some files is
ignored). Negative RGB values (out-of-gamut spectra) are clamped to zero,
matching the reference default `POWITACQ_CLIP_RGB = 1`.

# Format Specification
- Container: "tensor_file" binary format (little-endian), named tensor fields
- Required fields: theta_i, phi_i, ndf, sigma, vndf, rgb, luminance,
  description, jacobian
- Parameterization: half-vector warped through the material's own VNDF
  (data-adaptive, concentrates samples near specular features)
- Isotropic and anisotropic materials supported
"""

using StaticArrays
using LinearAlgebra
using Colors

# ============================================================================
# RGL Format Type
# ============================================================================

"""
    RGLFormat <: BRDFData

Format handler for RGL EPFL .bsdf tensor files.
"""
struct RGLFormat <: BRDFData end

const RGL_MAGIC = b"tensor_file\0"

# RGL RGB channels approximate the same display primaries as MERL's RGB
const RGL_WAVELENGTHS = Float32[630, 532, 465]

# dtype codes of the tensor_file container (only Float32/UInt8 are needed)
const RGL_DTYPE_UINT8 = 0x01
const RGL_DTYPE_FLOAT32 = 0x0a

# ============================================================================
# Format Detection
# ============================================================================

"""
    detect_format(::Type{RGLFormat}, filepath::String) -> Float64

Detect RGL .bsdf tensor files via the 12-byte "tensor_file" magic header.
"""
function detect_format(::Type{RGLFormat}, filepath::String)::Float64
    isfile(filepath) || return 0.0
    filesize(filepath) < 18 && return 0.0

    magic = open(filepath, "r") do io
        read(io, 12)
    end
    magic == RGL_MAGIC || return 0.0

    confidence = 0.9
    if endswith(lowercase(filepath), ".bsdf")
        confidence += 0.1
    end
    return confidence
end

# ============================================================================
# Tensor file container
# ============================================================================

struct RGLTensorField
    dtype::UInt8
    shape::Vector{Int}
    data::Vector{UInt8}   # raw bytes; reinterpret according to dtype
end

_rgl_type_size(dtype::UInt8) =
    dtype in (0x01, 0x02) ? 1 :
    dtype in (0x03, 0x04, 0x09) ? 2 :
    dtype in (0x05, 0x06, 0x0a) ? 4 :
    dtype in (0x07, 0x08, 0x0b) ? 8 :
    error("Invalid tensor file: unknown dtype $dtype")

"""
    _rgl_read_tensor(filepath) -> Dict{String, RGLTensorField}

Read all fields of a "tensor_file" container into memory.
"""
function _rgl_read_tensor(filepath::String)
    open(filepath, "r") do io
        magic = read(io, 12)
        magic == RGL_MAGIC || error("Invalid tensor file: bad magic header")
        version = read(io, 2)
        (version[1] == 1 && version[2] == 0) ||
            error("Invalid tensor file: unsupported version $(Int(version[1])).$(Int(version[2]))")
        n_fields = read(io, UInt32)

        fields = Dict{String, RGLTensorField}()
        for _ in 1:n_fields
            name_len = read(io, UInt16)
            name = String(read(io, Int(name_len)))
            ndim = read(io, UInt16)
            dtype = read(io, UInt8)
            offset = read(io, UInt64)
            shape = [Int(read(io, UInt64)) for _ in 1:ndim]

            total_bytes = _rgl_type_size(dtype) * prod(shape; init = 1)
            cur = position(io)
            seek(io, Int(offset))
            data = read(io, total_bytes)
            length(data) == total_bytes ||
                error("Invalid tensor file: truncated field \"$name\"")
            seek(io, cur)

            fields[name] = RGLTensorField(dtype, shape, data)
        end
        return fields
    end
end

function _rgl_float_field(fields, name::String)::Tuple{Vector{Float32}, Vector{Int}}
    haskey(fields, name) || error("RGL file is missing required field \"$name\"")
    f = fields[name]
    f.dtype == RGL_DTYPE_FLOAT32 ||
        error("RGL field \"$name\" has unexpected dtype (want Float32)")
    return collect(reinterpret(Float32, f.data)), f.shape
end

# ============================================================================
# Marginal2D warp (port of powitacq's Marginal2D<Dimension>)
# ============================================================================

"""
    RGLWarp2D

Piecewise-bilinear 2D density on [0,1]², optionally conditioned on up to
three extra parameters (linear interpolation between parameter slices).
Port of the reference loader's `Marginal2D<Dimension>`. Flat arrays keep the
exact 0-based row-major indexing of the C++ implementation; indices are
converted to 1-based only at the final array access.
"""
struct RGLWarp2D
    nx::Int
    ny::Int
    inv_patch_size_x::Float32
    inv_patch_size_y::Float32
    param_size::Vector{Int}
    param_strides::Vector{Int}
    param_values::Vector{Vector{Float32}}
    data::Vector{Float32}
    marginal_cdf::Vector{Float32}      # empty unless build_cdf
    conditional_cdf::Vector{Float32}   # empty unless build_cdf
end

"""
    _find_interval(size, pred) -> Int

Bisect an ordered set: largest 0-based index `i` (clamped to [0, size-2])
such that `pred(i)` holds. `pred` receives 0-based indices.
"""
function _find_interval(size::Int, pred::F) where {F}
    sz = size - 2
    first = 1
    while sz > 0
        half = sz >> 1
        middle = first + half
        if pred(middle)
            first = middle + 1
            sz -= half + 1
        else
            sz = half
        end
    end
    return clamp(first - 1, 0, size - 2)
end

function RGLWarp2D(nx::Int, ny::Int, data_in::AbstractVector{Float32},
                   param_size::Vector{Int}, param_values::Vector{Vector{Float32}};
                   normalize::Bool = true, build_cdf::Bool = true)
    build_cdf && !normalize &&
        error("RGLWarp2D: build_cdf implies normalize=true")

    ndim = length(param_size)
    param_strides = zeros(Int, ndim)
    slices = 1
    for i in ndim:-1:1
        param_size[i] >= 1 || error("RGLWarp2D: parameter resolution must be >= 1")
        param_strides[i] = param_size[i] > 1 ? slices : 0
        slices *= param_size[i]
    end

    n_values = nx * ny
    length(data_in) == slices * n_values ||
        error("RGLWarp2D: data size mismatch")

    data = Vector{Float32}(undef, slices * n_values)
    marginal_cdf = Float32[]
    conditional_cdf = Float32[]

    if build_cdf
        marginal_cdf = zeros(Float32, slices * ny)
        conditional_cdf = zeros(Float32, slices * n_values)

        for slice in 0:(slices - 1)
            dbase = slice * n_values   # 0-based base into data_in / cdfs
            mbase = slice * ny

            # Conditional CDF over columns, per row
            for y in 0:(ny - 1)
                sum = 0.0
                i = y * nx
                conditional_cdf[dbase + i + 1] = 0.0f0
                for x in 0:(nx - 2)
                    sum += 0.5 * (Float64(data_in[dbase + i + 1]) +
                                  Float64(data_in[dbase + i + 2]))
                    conditional_cdf[dbase + i + 2] = Float32(sum)
                    i += 1
                end
            end

            # Marginal CDF over rows
            marginal_cdf[mbase + 1] = 0.0f0
            sum = 0.0
            for y in 0:(ny - 2)
                sum += 0.5 * (Float64(conditional_cdf[dbase + (y + 1) * nx]) +
                              Float64(conditional_cdf[dbase + (y + 2) * nx]))
                marginal_cdf[mbase + y + 2] = Float32(sum)
            end

            # Normalize
            normalization = 1.0f0 / marginal_cdf[mbase + ny]
            for i in 1:n_values
                conditional_cdf[dbase + i] *= normalization
            end
            for i in 1:ny
                marginal_cdf[mbase + i] *= normalization
            end
            for i in 1:n_values
                data[dbase + i] = data_in[dbase + i] * normalization
            end
        end
    else
        for slice in 0:(slices - 1)
            dbase = slice * n_values
            normalization = 1.0f0 / ((nx - 1) * (ny - 1))
            if normalize
                sum = 0.0
                for y in 0:(ny - 2)
                    i = y * nx
                    for x in 0:(nx - 2)
                        v00 = data_in[dbase + i + 1]
                        v10 = data_in[dbase + i + 2]
                        v01 = data_in[dbase + i + nx + 1]
                        v11 = data_in[dbase + i + nx + 2]
                        sum += 0.25 * Float64(v00 + v10 + v01 + v11)
                        i += 1
                    end
                end
                normalization = Float32(1.0 / sum)
            end
            for i in 1:n_values
                data[dbase + i] = data_in[dbase + i] * normalization
            end
        end
    end

    return RGLWarp2D(nx, ny,
                     Float32(nx - 1), Float32(ny - 1),
                     param_size, param_strides, param_values,
                     data, marginal_cdf, conditional_cdf)
end

"""
Compute per-parameter interpolation weights into `pw` (2 per dimension) and
return the 0-based slice offset. `param` supplies one value per dimension.
"""
@inline function _param_weights!(pw::MVector{6, Float32}, w::RGLWarp2D, param)
    slice_offset = 0
    @inbounds for dim in 1:length(w.param_size)
        psize = w.param_size[dim]
        if psize == 1
            pw[2dim - 1] = 1.0f0
            pw[2dim] = 0.0f0
            continue
        end
        pv = w.param_values[dim]
        p = Float32(param[dim])
        idx = _find_interval(psize, i -> pv[i + 1] <= p)
        p0 = pv[idx + 1]
        p1 = pv[idx + 2]
        w1 = clamp((p - p0) / (p1 - p0), 0.0f0, 1.0f0)
        pw[2dim] = w1
        pw[2dim - 1] = 1.0f0 - w1
        slice_offset += w.param_strides[dim] * idx
    end
    return slice_offset
end

"""
Parameter-interpolated fetch: `arr[i0]` (0-based) blended across the 2^ndim
neighboring parameter slices, each `size` floats apart per stride unit.
"""
function _w2d_fetch(w::RGLWarp2D, arr::Vector{Float32}, dim::Int,
                    i0::Int, size::Int, pw::MVector{6, Float32})
    dim == 0 && return @inbounds arr[i0 + 1]
    i1 = i0 + w.param_strides[dim] * size
    @inbounds w0 = pw[2dim - 1]
    @inbounds w1 = pw[2dim]
    v0 = _w2d_fetch(w, arr, dim - 1, i0, size, pw)
    v1 = _w2d_fetch(w, arr, dim - 1, i1, size, pw)
    return muladd(v0, w0, v1 * w1)
end

"""
    _w2d_eval(w, pos, param) -> Float32

Evaluate the density at `pos` ∈ [0,1]² (bilinear interpolation), conditioned
on `param`. Port of `Marginal2D::eval`.
"""
function _w2d_eval(w::RGLWarp2D, pos::NTuple{2, Float32}, param)::Float32
    pw = MVector{6, Float32}(0, 0, 0, 0, 0, 0)
    slice_offset = _param_weights!(pw, w, param)
    ndim = length(w.param_size)

    px = pos[1] * w.inv_patch_size_x
    py = pos[2] * w.inv_patch_size_y
    ox = clamp(unsafe_trunc(Int, px), 0, w.nx - 2)
    oy = clamp(unsafe_trunc(Int, py), 0, w.ny - 2)

    w1x = px - ox;  w0x = 1.0f0 - w1x
    w1y = py - oy;  w0y = 1.0f0 - w1y

    size = w.nx * w.ny
    index = ox + oy * w.nx + slice_offset * size

    v00 = _w2d_fetch(w, w.data, ndim, index, size, pw)
    v10 = _w2d_fetch(w, w.data, ndim, index + 1, size, pw)
    v01 = _w2d_fetch(w, w.data, ndim, index + w.nx, size, pw)
    v11 = _w2d_fetch(w, w.data, ndim, index + w.nx + 1, size, pw)

    return muladd(w0y, muladd(w0x, v00, w1x * v10),
                  w1y * muladd(w0x, v01, w1x * v11)) *
           (w.inv_patch_size_x * w.inv_patch_size_y)
end

"""
    _w2d_invert(w, pos, param) -> (NTuple{2,Float32}, Float32)

Map a point on the warped domain back to the unit square (inverse of the
sample warp) and return it with the density at `pos`. Port of
`Marginal2D::invert`; requires the warp to have been built with CDFs.
"""
function _w2d_invert(w::RGLWarp2D, pos::NTuple{2, Float32},
                     param)::Tuple{NTuple{2, Float32}, Float32}
    pw = MVector{6, Float32}(0, 0, 0, 0, 0, 0)
    slice_offset = _param_weights!(pw, w, param)
    ndim = length(w.param_size)

    sx = pos[1] * w.inv_patch_size_x
    sy = pos[2] * w.inv_patch_size_y
    px = clamp(unsafe_trunc(Int, sx), 0, w.nx - 2)
    py = clamp(unsafe_trunc(Int, sy), 0, w.ny - 2)
    sx -= px
    sy -= py

    slice_size = w.nx * w.ny
    offset = px + py * w.nx + slice_offset * slice_size

    # Invert the X component
    v00 = _w2d_fetch(w, w.data, ndim, offset, slice_size, pw)
    v10 = _w2d_fetch(w, w.data, ndim, offset + 1, slice_size, pw)
    v01 = _w2d_fetch(w, w.data, ndim, offset + w.nx, slice_size, pw)
    v11 = _w2d_fetch(w, w.data, ndim, offset + w.nx + 1, slice_size, pw)

    w0x = 1.0f0 - sx;  w1x = sx
    w0y = 1.0f0 - sy;  w1y = sy

    c0 = muladd(w0y, v00, w1y * v01)
    c1 = muladd(w0y, v10, w1y * v11)
    pdf = muladd(w0x, c0, w1x * c1)

    sx *= c0 + 0.5f0 * sx * (c1 - c0)

    v0 = _w2d_fetch(w, w.conditional_cdf, ndim, offset, slice_size, pw)
    v1 = _w2d_fetch(w, w.conditional_cdf, ndim, offset + w.nx, slice_size, pw)
    sx += (1.0f0 - sy) * v0 + sy * v1

    offset = py * w.nx + slice_offset * slice_size
    r0 = _w2d_fetch(w, w.conditional_cdf, ndim, offset + w.nx - 1, slice_size, pw)
    r1 = _w2d_fetch(w, w.conditional_cdf, ndim, offset + 2 * w.nx - 1, slice_size, pw)
    sx /= (1.0f0 - sy) * r0 + sy * r1

    # Invert the Y component
    sy *= r0 + 0.5f0 * sy * (r1 - r0)
    moffset = py + slice_offset * w.ny
    sy += _w2d_fetch(w, w.marginal_cdf, ndim, moffset, w.ny, pw)

    return ((sx, sy), pdf * (w.inv_patch_size_x * w.inv_patch_size_y))
end

# ============================================================================
# RGL Internal Data Structure
# ============================================================================

"""
    RGLData

Internal storage for an RGL .bsdf material: the NDF, projected area (sigma),
VNDF warp, and RGB tensors of the adaptive parameterization.
"""
struct RGLData
    ndf::RGLWarp2D          # no params
    sigma::RGLWarp2D        # no params
    vndf::RGLWarp2D         # params: (phi_i, theta_i), with CDFs
    rgb::RGLWarp2D          # params: (phi_i, theta_i, channel)
    isotropic::Bool
    description::String
    vndf_res::Tuple{Int, Int}
    n_theta_i::Int
    n_phi_i::Int
end

function n_samples_str(d::RGLData)
    grid = d.isotropic ? "$(d.n_theta_i) θᵢ" : "$(d.n_phi_i)×$(d.n_theta_i) φᵢ×θᵢ"
    return "$grid × $(d.vndf_res[1])×$(d.vndf_res[2]) spec"
end

# ============================================================================
# Lookup Implementation
# ============================================================================

# Elevation-angle warp of the adaptive parameterization
@inline _rgl_theta2u(theta::Float32) = sqrt(theta * Float32(2 / π))
@inline _rgl_phi2u(phi::Float32) = (phi + Float32(π)) / Float32(2π)

# Numerically robust acos(d.z)
@inline function _rgl_elevation(d::SVector{3, Float32})::Float32
    return 2.0f0 * asin(0.5f0 * sqrt(d[1]^2 + d[2]^2 + (d[3] - 1.0f0)^2))
end

"""
    _brdf_value(data::RGLData, wavelengths, wi, wo) -> RGB{Float32}

Evaluate the measured BRDF (port of `powitacq_rgb::BRDF::eval`). The reference
`eval` returns `f_r * cos(theta_o)` (Mitsuba convention); we divide that factor
out so this returns the plain BRDF like every other format plugin — the
platform's renderers and fit residuals apply the cosine themselves. Without
this, `f_r * cos` breaks Helmholtz reciprocity and the universal LUT's phi_d
fold shows a visible seam along the plane of incidence.

Returns zero outside the upper hemisphere; negative values clamped to zero.
"""
function _brdf_value(data::RGLData,
                     wavelengths::Vector{Float32},
                     wi::SVector{3, Float32},
                     wo::SVector{3, Float32})::RGB{Float32}
    if wi[3] <= 0.0f0 || wo[3] <= 0.0f0
        return RGB{Float32}(0, 0, 0)
    end

    wm = normalize_vec(wi + wo)

    # Cartesian -> spherical coordinates
    theta_i = _rgl_elevation(wi)
    phi_i = atan(wi[2], wi[1])
    theta_m = _rgl_elevation(wm)
    phi_m = atan(wm[2], wm[1])

    # Spherical coordinates -> unit coordinate system
    u_wi = (_rgl_theta2u(theta_i), _rgl_phi2u(phi_i))
    u_wm_y = _rgl_phi2u(data.isotropic ? (phi_m - phi_i) : phi_m)
    u_wm = (_rgl_theta2u(theta_m), u_wm_y - floor(u_wm_y))

    params = (phi_i, theta_i)
    sample, _ = _w2d_invert(data.vndf, u_wm, params)

    r = max(0.0f0, _w2d_eval(data.rgb, sample, (phi_i, theta_i, 0.0f0)))
    g = max(0.0f0, _w2d_eval(data.rgb, sample, (phi_i, theta_i, 1.0f0)))
    b = max(0.0f0, _w2d_eval(data.rgb, sample, (phi_i, theta_i, 2.0f0)))

    scale = _w2d_eval(data.ndf, u_wm, params) /
            (4.0f0 * _w2d_eval(data.sigma, u_wi, params) * wo[3])
    (isfinite(scale) && scale > 0.0f0) || return RGB{Float32}(0, 0, 0)

    return RGB{Float32}(r * scale, g * scale, b * scale)
end

# ============================================================================
# Loader
# ============================================================================

function _rgl_failed(filepath::String, msg::String)
    dummy_warp = RGLWarp2D(2, 2, zeros(Float32, 4), Int[], Vector{Float32}[];
                           normalize = false, build_cdf = false)
    empty_data = RGLData(dummy_warp, dummy_warp, dummy_warp, dummy_warp,
                         true, "", (0, 0), 0, 0)
    return MeasuredBRDF(
        empty_data, RGL_WAVELENGTHS, basename(filepath);
        source_format = :rgl, status = :failed, error_msg = msg
    )
end

"""
    load_brdf(::Type{RGLFormat}, filepath::String) -> MeasuredBRDF

Load a BRDF from an RGL EPFL .bsdf tensor file.
"""
function load_brdf(::Type{RGLFormat}, filepath::String)::MeasuredBRDF
    isfile(filepath) || return _rgl_failed(filepath, "File not found: $filepath")

    try
        fields = _rgl_read_tensor(filepath)

        theta_i, theta_i_shape = _rgl_float_field(fields, "theta_i")
        phi_i, phi_i_shape = _rgl_float_field(fields, "phi_i")
        ndf, ndf_shape = _rgl_float_field(fields, "ndf")
        sigma, sigma_shape = _rgl_float_field(fields, "sigma")
        vndf, vndf_shape = _rgl_float_field(fields, "vndf")
        haskey(fields, "rgb") ||
            error("RGL file has no \"rgb\" field (spectral-only file? " *
                  "Use the RGB variant from the RGL database)")
        rgb_data, rgb_shape = _rgl_float_field(fields, "rgb")

        n_phi_i = phi_i_shape[1]
        n_theta_i = theta_i_shape[1]

        # Structural validation (mirrors the reference loader)
        (length(theta_i_shape) == 1 && length(phi_i_shape) == 1 &&
         length(ndf_shape) == 2 && length(sigma_shape) == 2 &&
         length(vndf_shape) == 4 &&
         vndf_shape[1] == n_phi_i && vndf_shape[2] == n_theta_i &&
         length(rgb_shape) == 5 &&
         rgb_shape[1] == n_phi_i && rgb_shape[2] == n_theta_i &&
         rgb_shape[3] == 3) ||
            error("Invalid RGL file structure (unexpected tensor shapes)")

        isotropic = n_phi_i <= 2
        if !isotropic
            reduction = round(Int, 2π / (phi_i[end] - phi_i[1]))
            reduction == 1 ||
                error("RGL file uses symmetry reduction (reduction=$reduction), not supported")
        end

        description = if haskey(fields, "description")
            String(copy(fields["description"].data))
        else
            ""
        end

        param_sizes = Int[n_phi_i, n_theta_i]
        param_vals = Vector{Float32}[phi_i, theta_i]

        rgl_data = RGLData(
            RGLWarp2D(ndf_shape[2], ndf_shape[1], ndf,
                      Int[], Vector{Float32}[];
                      normalize = false, build_cdf = false),
            RGLWarp2D(sigma_shape[2], sigma_shape[1], sigma,
                      Int[], Vector{Float32}[];
                      normalize = false, build_cdf = false),
            RGLWarp2D(vndf_shape[4], vndf_shape[3], vndf,
                      param_sizes, param_vals;
                      normalize = true, build_cdf = true),
            RGLWarp2D(rgb_shape[5], rgb_shape[4], rgb_data,
                      Int[n_phi_i, n_theta_i, 3],
                      Vector{Float32}[phi_i, theta_i, Float32[0, 1, 2]];
                      normalize = false, build_cdf = false),
            isotropic,
            description,
            (vndf_shape[4], vndf_shape[3]),
            n_theta_i,
            n_phi_i,
        )

        name = replace(basename(filepath), r"\.bsdf$"i => "")
        return MeasuredBRDF(
            rgl_data, RGL_WAVELENGTHS, name;
            source_format = :rgl, status = :ok, error_msg = nothing
        )
    catch e
        @error "Failed to load RGL file" exception=(e, catch_backtrace())
        return _rgl_failed(filepath, "Error reading file: $(sprint(showerror, e))")
    end
end

# ============================================================================
# Register Plugin
# ============================================================================

register_format!(RGLFormat)