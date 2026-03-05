"""
    BRDFPlatform

A plugin-based platform for analyzing and fitting BRDF models to measured data.

# Main Features
- Load BRDF data from multiple formats (MERL, BiRD, etc.)
- Fit analytical models (Lambertian, Cook-Torrance, Ward, etc.)
- Interactive visualization (WGLMakie + Three.js)
- Extensible plugin system

# Quick Start
```julia
using BRDFPlatform

# Load BRDF data
data = load_brdf("path/to/file.binary")

# Fit a model
model = fit_brdf(LevenbergMarquardt(), CookTorrance, data)

# Visualize
run_app()
```
"""
module BRDFPlatform

# Load core types first (included directly in this module)
include("types.jl")

# ============================================================================
# Plugin Discovery and Loading
# ============================================================================

"""
    discover_plugins(plugin_dir::String, plugin_type::Symbol)

Discover and load plugin files from a directory.
Returns vector of successfully loaded plugin file paths.

# Arguments
- `plugin_dir`: Directory to scan for .jl files
- `plugin_type`: One of :format, :model, :optimizer
"""
function discover_plugins(plugin_dir::String, plugin_type::Symbol)
    if !isdir(plugin_dir)
        @warn "Plugin directory not found: $plugin_dir"
        return String[]
    end

    loaded_plugins = String[]

    for file in readdir(plugin_dir)
        if endswith(file, ".jl")
            filepath = joinpath(plugin_dir, file)
            try
                include(filepath)
                push!(loaded_plugins, filepath)
                @info "Loaded $plugin_type plugin: $file"
            catch e
                @error "Failed to load $plugin_type plugin: $file" exception=(e, catch_backtrace())
            end
        end
    end

    return loaded_plugins
end

"""
    load_core_plugins()

Load all built-in plugins from src/ subdirectories.
Called automatically on module initialization.
"""
function load_core_plugins()
    src_dir = @__DIR__

    # Load format plugins
    format_dir = joinpath(src_dir, "formats")
    discover_plugins(format_dir, :format)

    # Load model plugins
    model_dir = joinpath(src_dir, "models")
    discover_plugins(model_dir, :model)

    # Load optimizer plugins
    optimizer_dir = joinpath(src_dir, "optimizers")
    discover_plugins(optimizer_dir, :optimizer)

    # Load visualization module
    viz_file = joinpath(src_dir, "visualization", "plots.jl")
    if isfile(viz_file)
        try
            include(viz_file)
            @info "Loaded visualization module"
        catch e
            @error "Failed to load visualization module" exception=(e, catch_backtrace())
        end
    end

    return nothing
end

"""
    load_user_plugins()

Load user-provided plugins from ~/.brdf_platform/ directories.
Call this to reload user plugins after editing.
"""
function load_user_plugins()
    user_dir = joinpath(homedir(), ".brdf_platform")

    if !isdir(user_dir)
        @info "Creating user plugin directory: $user_dir"
        mkpath(joinpath(user_dir, "formats"))
        mkpath(joinpath(user_dir, "models"))
        mkpath(joinpath(user_dir, "optimizers"))
    end

    # Load user plugins
    discover_plugins(joinpath(user_dir, "formats"), :format)
    discover_plugins(joinpath(user_dir, "models"), :model)
    discover_plugins(joinpath(user_dir, "optimizers"), :optimizer)

    return nothing
end

# ============================================================================
# High-Level API Functions
# ============================================================================

"""
    detect_format(filepath::String) -> Union{Type{<:BRDFData}, Nothing}

Auto-detect the format of a BRDF file by trying all registered format plugins.
Returns the format type with highest confidence (>0.6), or Nothing if no match.
"""
function detect_format(filepath::String)
    if !isfile(filepath)
        @error "File not found: $filepath"
        return nothing
    end

    best_format = nothing
    best_confidence = 0.6  # Minimum threshold

    for (name, format_type) in FORMAT_REGISTRY
        try
            if hasmethod(detect_format, (Type{format_type}, String))
                confidence = detect_format(format_type, filepath)
                if confidence > best_confidence
                    best_confidence = confidence
                    best_format = format_type
                end
            end
        catch e
            @debug "Format detection failed for $name" exception=e
        end
    end

    if best_format === nothing
        @warn "Could not detect format for: $filepath"
    else
        @info "Detected format: $(nameof(best_format)) (confidence: $(round(best_confidence, digits=2)))"
    end

    return best_format
end

"""
    load_brdf(filepath::String) -> MeasuredBRDF

Load BRDF data from a file, auto-detecting the format.
Falls back to manual format detection if auto-detection fails.
"""
function load_brdf(filepath::String)
    format_type = detect_format(filepath)

    if format_type === nothing
        # Return failed MeasuredBRDF (use empty data - format doesn't matter for failed state)
        empty_data = nothing
        return MeasuredBRDF(
            empty_data,
            Float32[630, 532, 465],
            basename(filepath),
            source_format = :unknown,
            status = :failed,
            error_msg = "Could not detect file format"
        )
    end

    try
        return load_brdf(format_type, filepath)
    catch e
        @error "Failed to load BRDF data" exception=(e, catch_backtrace())
        empty_data = nothing
        return MeasuredBRDF(
            empty_data,
            Float32[630, 532, 465],
            basename(filepath),
            source_format = Symbol(lowercase(string(nameof(format_type)))),
            status = :failed,
            error_msg = "Error loading file: $(sprint(showerror, e))"
        )
    end
end

"""
    list_plugins() -> Dict{Symbol, Vector{Symbol}}

List all registered plugins by category.
"""
function list_plugins()
    return Dict(
        :formats => collect(keys(FORMAT_REGISTRY)),
        :models => collect(keys(MODEL_REGISTRY)),
        :optimizers => collect(keys(OPTIMIZER_REGISTRY))
    )
end

# ============================================================================
# Web Application
# ============================================================================

# Load apps last (after all plugins are defined)
include("app/main_app.jl")
include("app/interactive_app.jl")

# ============================================================================
# Module Initialization
# ============================================================================

function __init__()
    @info "Initializing BRDFPlatform..."
    load_core_plugins()
    load_user_plugins()

    # Print summary
    plugins = list_plugins()
    @info "Loaded plugins:" formats=length(plugins[:formats]) models=length(plugins[:models]) optimizers=length(plugins[:optimizers])
end

# ============================================================================
# Exports
# ============================================================================

# Re-export types
export BRDFData, BRDFModel, BRDFOptimizer
export MeasuredBRDF
export vec3, normalize_vec, rgb

# Model interface
export evaluate, glsl_shader, shader_parameters

# Concrete model types (loaded as plugins, but re-exported for convenience)
export LambertianModel, CookTorranceModel, GGXModel

# BRDF lookup interface
export lookup, lookup_batch, brdf_value

# Plugin functions
export register_format!, register_model!, register_optimizer!
export load_user_plugins, list_plugins

# High-level API
export detect_format, load_brdf

# Web application
export run_app, run_interactive_app

# Visualization
export create_hemisphere_plot, create_simple_hemisphere_plot
export create_polar_slice_plot, create_simple_polar_slice_plot

end # module BRDFPlatform
