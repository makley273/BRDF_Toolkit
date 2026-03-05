# BRDF Analysis Platform - Architecture Design

**Date:** January 2025  
**Principle:** KISS - Keep it Simple, Stupid  
**Status:** Design Phase

---

## 1. Module Structure

### Directory Layout (Flat & Scannable)

```
src/
├── BRDFPlatform.jl          # Main module, plugin discovery, exports
│
├── types.jl                  # Abstract types for plugin system
│
├── formats/                  # Input format plugins
│   ├── merl.jl              # MERL binary format
│   └── bird.jl              # Universal BiRD format
│
├── models/                   # BRDF model plugins  
│   ├── lambertian.jl
│   ├── cook_torrance.jl
│   └── ward.jl
│
├── optimizers/               # Fitting algorithm plugins
│   ├── levenberg_marquardt.jl
│   └── nelder_mead.jl
│
├── rendering/                # Three.js + GLSL integration
│   ├── threejs_renderer.jl  # Bonito/Three.js bridge
│   └── shaders/             # GLSL shader files
│       ├── fallback.glsl    # Table-based fallback
│       ├── lambertian.glsl
│       └── cook_torrance.glsl
│
├── visualization/            # Scientific visualization
│   └── plots.jl             # WGLMakie hemisphere plots, heatmaps
│
└── app/                      # Web interface
    └── main_app.jl          # Bonito UI composition
```

**Rationale:** Everything visible at one level. No nested module hierarchies. Each file is self-contained and implements one plugin.

---

## 2. Type System (Common Base for Interchangeability)

### Core Abstract Types

```julia
# types.jl

"""
Base type for all BRDF data formats.
Ensures format plugins can be used interchangeably.
"""
abstract type BRDFData end

"""
Base type for all BRDF models (analytical & neural).
Ensures model plugins can be used interchangeably for plotting/comparison.
"""
abstract type BRDFModel end

"""
Base type for all optimization algorithms.
Ensures optimizer plugins can be used interchangeably.
"""
abstract type BRDFOptimizer end
```

### Concrete Data Structure

```julia
"""
Internal representation - all formats convert to this.
Measured BRDF data with metadata.

Sampling knowledge is encapsulated within each format's internal data type
(e.g. MERLData knows its Rusinkiewicz grid dimensions). There is no separate
SamplingPattern hierarchy — each format type handles its own lookup logic.
"""
struct MeasuredBRDF <: BRDFData
    # Internal storage (format-specific, opaque).
    # Each format type (e.g. MERLData) encapsulates its own sampling knowledge.
    _data::Any

    # Wavelength information
    wavelengths::Vector{Float32}  # nm, e.g., [630, 532, 465] for RGB

    # Metadata
    name::String
    source_format::Symbol        # :merl, :bird, :utia

    # Error handling
    status::Symbol               # :ok, :partial, :failed
    error_msg::Union{Nothing, String}
end
```

**Why no separate SamplingPattern?** Sampling is an implementation detail of each format. `MERLData` already stores its grid dimensions (`n_theta_h`, `n_theta_d`, `n_phi_d`) and uses them for lookup/interpolation internally. Exposing a parallel `SamplingPattern` hierarchy would duplicate information and leak format internals into the public interface. The `brdf_value(brdf, wi, wo)` function dispatches on `typeof(brdf._data)` — callers never need to know the sampling strategy.

---

## 3. Plugin System (Auto-Discovery + Common Interface)

### Format Plugins (with Auto-Detection)

```julia
# Common interface that each format plugin must implement

"""
Check if file matches this format.
Return confidence score 0.0-1.0 (for auto-detection).
"""
function detect_format(::Type{<:BRDFData}, path::String)::Float64
    error("Not implemented")
end

"""
Load BRDF data from file.
"""
function load_brdf(::Type{<:BRDFData}, path::String)::MeasuredBRDF
    error("Not implemented")
end
```

**Example implementation:**

```julia
# formats/merl.jl

struct MERLFormat <: BRDFData end

function detect_format(::Type{MERLFormat}, path::String)::Float64
    try
        open(path, "r") do io
            # MERL files have specific structure:
            # - Exactly 90 × 90 × 180 samples = 1,458,000 entries
            # - 3 channels (RGB) × 4 bytes (Float32) = 12 bytes per sample
            # - Total: 1,458,000 × 3 × 4 = 17,496,000 bytes
            size = filesize(path)
            if size == 17_496_000
                return 1.0  # Definitely MERL
            elseif endswith(lowercase(path), ".binary")
                # Possible MERL but wrong size
                return 0.6
            else
                return 0.0
            end
        end
    catch
        return 0.0
    end
end

function load_brdf(::Type{MERLFormat}, path::String)::MeasuredBRDF
    # Read MERL binary format
    data = open(path, "r") do io
        # Read as [RGB × theta_h × theta_d × phi_d]
        raw = Array{Float32}(undef, 3, 90, 90, 180)
        read!(io, raw)
        raw
    end

    # MERLData encapsulates sampling knowledge (Rusinkiewicz grid dimensions)
    merl_data = MERLData(data, 90, 90, 180)

    return MeasuredBRDF(
        merl_data,
        Float32[630, 532, 465],  # RGB approximation
        splitext(basename(path))[1],
        source_format = :merl,
        status = :ok,
        error_msg = nothing
    )
end

# Register plugin
register_format!(MERLFormat)
```

**BiRD format detection:**

```julia
# formats/bird.jl

struct BiRDFormat <: BRDFData end

function detect_format(::Type{BiRDFormat}, path::String)::Float64
    try
        open(path, "r") do io
            # Check if it's valid JSON and has BiRD structure
            content = String(read(io, 512))  # Read first 512 bytes
            
            if occursin("\"brdf\"", content) && 
               (endswith(lowercase(path), ".bird") || 
                endswith(lowercase(path), ".json"))
                return 0.9  # Likely BiRD
            elseif endswith(lowercase(path), ".bird")
                return 0.7  # Extension matches but content unclear
            else
                return 0.0
            end
        end
    catch
        return 0.0
    end
end
```

### Model Plugins (Dual Implementation)

```julia
# Common interface for BRDF models

"""
Evaluate BRDF model in Julia (for fitting, error computation).
"""
function evaluate(model::BRDFModel, wi::Vec3, wo::Vec3)::RGB{Float32}
    error("Not implemented")
end

"""
Return GLSL shader code for this model (for real-time rendering).
Returns: (shader_source::String, uniform_names::Vector{String})
"""
function glsl_shader(model::BRDFModel)::Tuple{String, Vector{String}}
    error("Not implemented")
end

"""
Extract shader parameters from fitted model.
Returns: Dict{String, Any} mapping uniform names to values.
"""
function shader_parameters(model::BRDFModel)::Dict{String, Any}
    error("Not implemented")
end
```

**Example implementation:**

```julia
# models/cook_torrance.jl

struct CookTorranceModel <: BRDFModel
    # Fitted parameters
    roughness::Float32
    F0::Float32  # Fresnel reflectance at normal incidence
    albedo::RGB{Float32}
end

function evaluate(m::CookTorranceModel, wi::Vec3, wo::Vec3)::RGB{Float32}
    # Cook-Torrance BRDF evaluation in Julia
    # ... math implementation
end

function glsl_shader(m::CookTorranceModel)
    shader = """
    vec3 cook_torrance(vec3 wi, vec3 wo, float roughness, float F0, vec3 albedo) {
        // Cook-Torrance BRDF in GLSL
        // ... same math as Julia version
    }
    """
    
    uniforms = ["u_roughness", "u_F0", "u_albedo"]
    return (shader, uniforms)
end

function shader_parameters(m::CookTorranceModel)
    return Dict(
        "u_roughness" => m.roughness,
        "u_F0" => m.F0,
        "u_albedo" => [m.albedo.r, m.albedo.g, m.albedo.b]
    )
end
```

### Optimizer Plugins

```julia
# Common interface for optimizers

"""
Fit BRDF model to measured data.
Returns: fitted model instance
"""
function fit_brdf(
    optimizer::BRDFOptimizer,
    model_type::Type{<:BRDFModel},
    data::MeasuredBRDF
)::BRDFModel
    error("Not implemented")
end
```

**Example implementation:**

```julia
# optimizers/levenberg_marquardt.jl

struct LevenbergMarquardt <: BRDFOptimizer
    max_iterations::Int
    tolerance::Float64
end

function fit_brdf(
    opt::LevenbergMarquardt,
    ::Type{CookTorranceModel},
    data::MeasuredBRDF
)::CookTorranceModel
    # LM optimization implementation
    # ...
    
    return CookTorranceModel(
        roughness = fitted_roughness,
        F0 = fitted_F0,
        albedo = fitted_albedo
    )
end
```

---

## 4. Plugin Discovery & Registration

### Explicit Registration System

```julia
# BRDFPlatform.jl

# Plugin registries
const FORMAT_PLUGINS = Dict{String, Type{<:BRDFData}}()
const MODEL_PLUGINS = Dict{String, Type{<:BRDFModel}}()
const OPTIMIZER_PLUGINS = Dict{String, Type{<:BRDFOptimizer}}()

"""
Register a format plugin. Called explicitly at end of each format file.
"""
function register_format!(T::Type{<:BRDFData})
    name = lowercase(string(nameof(T)))
    FORMAT_PLUGINS[name] = T
    @info "Registered format plugin: $name"
    return T
end

"""
Register a model plugin. Called explicitly at end of each model file.
"""
function register_model!(T::Type{<:BRDFModel})
    name = lowercase(string(nameof(T)))
    MODEL_PLUGINS[name] = T
    @info "Registered model plugin: $name"
    return T
end

"""
Register an optimizer plugin. Called explicitly at end of each optimizer file.
"""
function register_optimizer!(T::Type{<:BRDFOptimizer})
    name = lowercase(string(nameof(T)))
    OPTIMIZER_PLUGINS[name] = T
    @info "Registered optimizer plugin: $name"
    return T
end

"""
Load all plugins by including files in directory.
Files must call register_*!() at the end.
"""
function load_plugins!(dir::String)
    for file in readdir(dir; join=true)
        if endswith(file, ".jl")
            include(file)
        end
    end
end

# At module initialization
load_plugins!(joinpath(@__DIR__, "formats"))
load_plugins!(joinpath(@__DIR__, "models"))
load_plugins!(joinpath(@__DIR__, "optimizers"))
```

### Auto-Detection for Format Loading

```julia
"""
Load BRDF file with automatic format detection.
Tries all registered format plugins and uses the one with highest confidence.
Requires minimum confidence of 0.6 to avoid false positives.
"""
function load_brdf_auto(path::String)::MeasuredBRDF
    if !isfile(path)
        return MeasuredBRDF(
            nothing,
            Float32[630, 532, 465],
            "",
            source_format = :unknown,
            status = :failed,
            error_msg = "File not found: $path"
        )
    end
    
    # Try all format plugins
    scores = Dict{Type, Float64}()
    
    for (name, format_type) in FORMAT_PLUGINS
        try
            confidence = detect_format(format_type, path)
            if confidence > 0.0
                scores[format_type] = confidence
                @debug "Format $name confidence: $confidence"
            end
        catch e
            @warn "Format detection failed for $name" exception=e
        end
    end
    
    # Check if we have any candidates
    if isempty(scores)
        return MeasuredBRDF(
            nothing,
            Float32[630, 532, 465],
            basename(path),
            source_format = :unknown,
            status = :failed,
            error_msg = "No format plugin could identify this file"
        )
    end
    
    # Find best match
    best_format = nothing
    best_confidence = 0.0
    
    for (format_type, confidence) in scores
        if confidence > best_confidence
            best_confidence = confidence
            best_format = format_type
        end
    end
    
    # Require minimum confidence threshold
    MIN_CONFIDENCE = 0.6
    if best_confidence < MIN_CONFIDENCE
        format_names = join(["$k: $(round(v, digits=2))" for (k,v) in scores], ", ")
        return MeasuredBRDF(
            nothing,
            Float32[630, 532, 465],
            basename(path),
            source_format = :unknown,
            status = :failed,
            error_msg = "Uncertain format (best: $best_confidence < $MIN_CONFIDENCE). Scores: $format_names"
        )
    end
    
    # Load with best matching format
    try
        @info "Loading $path as $(best_format) (confidence: $best_confidence)"
        return load_brdf(best_format, path)
    catch e
        return MeasuredBRDF(
            nothing,
            Float32[630, 532, 465],
            basename(path),
            source_format = :unknown,
            status = :failed,
            error_msg = "Load failed with $(best_format): $(sprint(showerror, e))"
        )
    end
end
```

---

## 5. Fallback Strategy (Two Layers)

### Layer 1: GLSL Table-Based Fallback

```glsl
// shaders/fallback.glsl

// Fallback shader: direct lookup from measured data texture
// No model evaluation - just shows raw measurements

uniform sampler2D u_brdf_table;  // Pre-computed BRDF lookup table

vec3 evaluate_brdf_fallback(vec3 wi, vec3 wo) {
    // Convert angles to texture coordinates
    // (implementation depends on sampling pattern)
    vec2 uv = angles_to_texture_coords(wi, wo);
    
    // Direct texture lookup
    vec3 brdf_value = texture(u_brdf_table, uv).rgb;
    
    return brdf_value;
}
```

**When to use:** When model shader fails to compile or encounters errors.

### Layer 2: Error Handling & Recovery

```julia
# rendering/threejs_renderer.jl

"""
Render BRDF with automatic fallback if model shader fails.
"""
function render_brdf(
    session::Session,
    container::DOM.Node,
    model::BRDFModel,
    data::MeasuredBRDF
)
    # Try model-specific shader first
    try
        shader_code, uniforms = glsl_shader(model)
        params = shader_parameters(model)
        
        # Send to Three.js
        onjs(session, container, js"""
            // Try to compile and use model shader
            try {
                setupModelShader($shader_code, $params);
            } catch(e) {
                console.error("Model shader failed:", e);
                // Fall back to table-based rendering
                setupFallbackShader($data);
            }
        """)
    catch e
        @warn "GLSL shader generation failed, using fallback" exception=e
        
        # Use fallback table-based shader
        onjs(session, container, js"""
            setupFallbackShader($data);
        """)
    end
end
```

**Recovery path:** Model shader → Table-based shader → Clear error message

---

## 6. Hot Reload Support (User Plugin Development)

### Design Philosophy

Users can add their own plugins by dropping `.jl` files in:
- `~/.brdf_platform/formats/`
- `~/.brdf_platform/models/`
- `~/.brdf_platform/optimizers/`

In **development mode**, Revise.jl handles automatic reloading of package code. For **user plugins**, we use a simple manual reload mechanism triggered by UI button or file watching.

### Implementation

```julia
# BRDFPlatform.jl

# User plugin directories
const USER_PLUGIN_DIRS = (
    formats = expanduser("~/.brdf_platform/formats"),
    models = expanduser("~/.brdf_platform/models"),
    optimizers = expanduser("~/.brdf_platform/optimizers")
)

"""
Load user-provided plugins from home directory.
Called at initialization and when user clicks "Reload Plugins" in UI.
"""
function load_user_plugins!()
    mkpath(USER_PLUGIN_DIRS.formats)
    mkpath(USER_PLUGIN_DIRS.models)
    mkpath(USER_PLUGIN_DIRS.optimizers)
    
    # Clear user plugins (not built-in ones)
    # We track which are user plugins via metadata
    filter!(p -> !get(p.second.metadata, :user_plugin, false), FORMAT_PLUGINS)
    filter!(p -> !get(p.second.metadata, :user_plugin, false), MODEL_PLUGINS)
    filter!(p -> !get(p.second.metadata, :user_plugin, false), OPTIMIZER_PLUGINS)
    
    # Load user plugins
    try
        for file in readdir(USER_PLUGIN_DIRS.formats; join=true)
            endswith(file, ".jl") && include(file)
        end
        for file in readdir(USER_PLUGIN_DIRS.models; join=true)
            endswith(file, ".jl") && include(file)
        end
        for file in readdir(USER_PLUGIN_DIRS.optimizers; join=true)
            endswith(file, ".jl") && include(file)
        end
        
        @info "User plugins loaded successfully"
    catch e
        @error "Error loading user plugins" exception=e
    end
end

# Call at module init
function __init__()
    load_user_plugins!()
end

"""
Mark a plugin as user-provided (for reload management).
User plugin files should call this instead of plain register_*!()
"""
register_user_format!(T) = (register_format!(T); 
                            FORMAT_PLUGINS[lowercase(string(nameof(T)))].metadata[:user_plugin] = true; 
                            T)
register_user_model!(T) = (register_model!(T); 
                           MODEL_PLUGINS[lowercase(string(nameof(T)))].metadata[:user_plugin] = true; 
                           T)
register_user_optimizer!(T) = (register_optimizer!(T); 
                               OPTIMIZER_PLUGINS[lowercase(string(nameof(T)))].metadata[:user_plugin] = true; 
                               T)
```

### UI Integration

```julia
# In main Bonito app

reload_button = Button("🔄 Reload User Plugins")

on(reload_button) do click
    try
        load_user_plugins!()
        notification[] = "✓ Plugins reloaded"
    catch e
        notification[] = "✗ Reload failed: $(sprint(showerror, e))"
    end
end
```

### Example User Plugin

```julia
# ~/.brdf_platform/models/my_custom_brdf.jl

using BRDFPlatform

struct MyCustomBRDF <: BRDFModel
    param1::Float32
    param2::Float32
end

function BRDFPlatform.evaluate(m::MyCustomBRDF, wi::Vec3, wo::Vec3)::RGB{Float32}
    # Your custom BRDF math here
    # ...
end

function BRDFPlatform.glsl_shader(m::MyCustomBRDF)
    shader = """
    vec3 my_custom_brdf(vec3 wi, vec3 wo, float param1, float param2) {
        // Your GLSL implementation
        // ...
    }
    """
    return (shader, ["u_param1", "u_param2"])
end

function BRDFPlatform.shader_parameters(m::MyCustomBRDF)
    return Dict("u_param1" => m.param1, "u_param2" => m.param2)
end

# IMPORTANT: Use register_user_*! for user plugins
BRDFPlatform.register_user_model!(MyCustomBRDF)
```

**Note:** Package development uses Revise.jl automatically. User plugins use manual reload button. This avoids world-age issues and keeps it simple.

---

## 7. Data Flow Architecture

### High-Level Flow

```
User uploads file
    ↓
Auto-detect format → Load with appropriate plugin
    ↓
Convert to MeasuredBRDF (internal representation)
    ↓
Select models to fit (user choice or auto-loop)
    ↓
For each model:
    Fit parameters (Julia, using optimizer plugin)
    Generate GLSL shader
    Compute error metrics (Julia)
    ↓
Visualization:
    - WGLMakie: Hemisphere plots, error heatmaps (Julia)
    - Three.js: Material rendering with GLSL shaders (browser)
```

### Three.js Integration (Bonito Bridge)

**Key Principle:** Three.js runs entirely in browser. Julia only sends:
1. GLSL shader code (once per model)
2. Uniform values (when parameters change)
3. Lookup table texture (for fallback rendering)

```julia
# rendering/threejs_renderer.jl

"""
Create Three.js material renderer container.
Returns DOM node that hosts the WebGL canvas.
"""
function create_threejs_renderer(session::Session, width=400, height=400)
    container = DOM.div(style=Styles("width" => "$(width)px", "height" => "$(height)px"))
    
    # Initialize Three.js scene
    onload(session, container, js"""
    function(container) {
        // Import Three.js
        const THREE = await import('https://cdn.skypack.dev/three@0.140.0');
        
        // Create scene
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(45, $width / $height, 0.1, 100);
        camera.position.set(0, 0, 3);
        
        // Create renderer
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize($width, $height);
        container.appendChild(renderer.domElement);
        
        // Create sphere geometry for BRDF visualization
        const geometry = new THREE.SphereGeometry(1, 64, 64);
        
        // Initial material (will be replaced)
        const material = new THREE.MeshBasicMaterial({ color: 0x808080 });
        const sphere = new THREE.Mesh(geometry, material);
        scene.add(sphere);
        
        // Lighting
        const light = new THREE.DirectionalLight(0xffffff, 1.0);
        light.position.set(1, 1, 1);
        scene.add(light);
        
        // Render loop
        function animate() {
            requestAnimationFrame(animate);
            sphere.rotation.y += 0.005;
            renderer.render(scene, camera);
        }
        animate();
        
        // Store references for later updates
        container.three_scene = { scene, camera, renderer, sphere, geometry };
    }
    """)
    
    return container
end

"""
Update Three.js material with BRDF model shader.
"""
function update_brdf_material(
    session::Session,
    container::DOM.Node,
    model::BRDFModel
)
    shader_code, uniform_names = glsl_shader(model)
    params = shader_parameters(model)
    
    # Convert params to JS-friendly format
    uniforms_js = Dict{String, Any}()
    for (name, value) in params
        if value isa AbstractVector && length(value) == 3
            # RGB color → THREE.Vector3
            uniforms_js[name] = Dict("value" => value)
        elseif value isa Number
            # Scalar
            uniforms_js[name] = Dict("value" => value)
        end
    end
    
    # Send shader + uniforms to browser
    onjs(session, container, js"""
    function(container) {
        const three = container.three_scene;
        if (!three) {
            console.error("Three.js not initialized yet");
            return;
        }
        
        try {
            // Create custom ShaderMaterial
            const material = new THREE.ShaderMaterial({
                uniforms: $(uniforms_js),
                vertexShader: `
                    varying vec3 vNormal;
                    varying vec3 vPosition;
                    
                    void main() {
                        vNormal = normalize(normalMatrix * normal);
                        vPosition = position;
                        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                    }
                `,
                fragmentShader: `
                    varying vec3 vNormal;
                    varying vec3 vPosition;
                    
                    // User's BRDF shader code inserted here
                    $(shader_code)
                    
                    void main() {
                        vec3 normal = normalize(vNormal);
                        vec3 viewDir = normalize(cameraPosition - vPosition);
                        vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
                        
                        // Call user's BRDF function
                        // Function name is extracted from shader_code
                        vec3 color = cook_torrance(lightDir, viewDir, 
                                                   u_roughness, u_F0, u_albedo);
                        
                        gl_FragColor = vec4(color, 1.0);
                    }
                `
            });
            
            // Update sphere material
            three.sphere.material.dispose(); // Clean up old material
            three.sphere.material = material;
            
        } catch(e) {
            console.error("Shader compilation failed:", e);
            // Fall back to error material (solid red)
            three.sphere.material = new THREE.MeshBasicMaterial({ color: 0xff0000 });
        }
    }
    """)
end

"""
Update just the uniform values (fast, no shader recompilation).
"""
function update_brdf_uniforms(
    session::Session,
    container::DOM.Node,
    params::Dict{String, Any}
)
    onjs(session, container, js"""
    function(container) {
        const three = container.three_scene;
        if (!three || !three.sphere.material.uniforms) return;
        
        const uniforms = three.sphere.material.uniforms;
        const params = $(params);
        
        for (const [name, value] of Object.entries(params)) {
            if (uniforms[name]) {
                uniforms[name].value = value;
            }
        }
    }
    """)
end
```

### Usage Example in Main App

```julia
# app/main_app.jl

App() do session
    # Create renderer once
    threejs_container = create_threejs_renderer(session, 400, 400)
    
    # Fit button
    fit_button = Button("Fit Cook-Torrance")
    
    on(fit_button) do click
        # Fit model in Julia
        model = fit_brdf(
            LevenbergMarquardt(max_iterations=100),
            CookTorranceModel,
            loaded_data[]
        )
        
        # Update Three.js visualization
        update_brdf_material(session, threejs_container, model)
    end
    
    # Parameter sliders for interactive adjustment
    roughness_slider = Slider(0.01:0.01:1.0)
    
    on(roughness_slider) do roughness
        # Update just this uniform (fast)
        update_brdf_uniforms(session, threejs_container, 
                            Dict("u_roughness" => roughness))
    end
    
    return Grid(
        fit_button,
        roughness_slider,
        threejs_container;
        columns="1fr"
    )
end
```

**Performance notes:**
- Shader compilation happens once per model switch
- Uniform updates are real-time (60 FPS)
- No Julia → JS communication during rotation/animation


---

## 8. Error Handling Philosophy

### Graceful Degradation

```julia
"""
Try operation with fallback chain.
Returns: (success::Bool, result::Any, error::Union{Nothing, Exception})
"""
function try_with_fallback(primary::Function, fallback::Function)
    try
        result = primary()
        return (true, result, nothing)
    catch e
        @warn "Primary operation failed, using fallback" exception=e
        try
            result = fallback()
            return (false, result, e)  # Used fallback
        catch e2
            @error "Fallback also failed!" exception=e2
            return (false, nothing, e2)
        end
    end
end

# Usage example
success, model, error = try_with_fallback(
    () -> fit_brdf(LevenbergMarquardt(), CookTorranceModel, data),
    () -> fit_brdf(NelderMead(), CookTorranceModel, data)
)

if !success
    # Show error to user, but app still works (maybe with table visualization)
end
```

### User-Visible Errors

```julia
# In Bonito UI
error_display = Observable{Union{Nothing, String}}(nothing)

error_card = map(error_display) do msg
    if isnothing(msg)
        return DOM.div()  # No error
    else
        return Card(
            DOM.div(
                "⚠️ ", msg,
                style=Styles("color" => "red", "background-color" => "#fee")
            )
        )
    end
end
```

**Principle:** Never crash. Always provide feedback. Degrade gracefully.

---

## 9. Validation Testing Framework

### Critical Need: Julia vs GLSL Verification

Every BRDF model exists in two implementations. **We must verify they produce identical results.**

### Test Structure

```julia
# test/validation.jl

using Test
using BRDFPlatform

"""
Compare Julia and GLSL implementations of a BRDF model.
Returns maximum relative error.
"""
function validate_model(model::BRDFModel; n_samples=1000, tolerance=1e-4)
    # Generate random test directions
    Random.seed!(42)
    test_cases = [(
        wi = normalize(Vec3(rand(3)...)),
        wo = normalize(Vec3(rand(3)...))
    ) for _ in 1:n_samples]
    
    # Evaluate in Julia
    julia_results = map(test_cases) do tc
        evaluate(model, tc.wi, tc.wo)
    end
    
    # Evaluate in GLSL (via browser automation)
    glsl_results = evaluate_glsl_batch(model, test_cases)
    
    # Compare
    errors = map(julia_results, glsl_results) do jval, gval
        # Relative error per channel
        max_rel_error = maximum(abs.((jval .- gval) ./ (jval .+ 1e-8)))
        return max_rel_error
    end
    
    max_error = maximum(errors)
    
    @test max_error < tolerance
    
    if max_error >= tolerance
        @warn "Model validation failed!" model max_error tolerance
        # Print worst cases for debugging
        worst_idx = argmax(errors)
        @show test_cases[worst_idx]
        @show julia_results[worst_idx]
        @show glsl_results[worst_idx]
    end
    
    return max_error
end

"""
Evaluate GLSL shader on batch of inputs.
Uses headless browser (Playwright.jl or similar).
"""
function evaluate_glsl_batch(model::BRDFModel, test_cases)
    shader_code, uniform_names = glsl_shader(model)
    params = shader_parameters(model)
    
    # Launch headless browser with WebGL
    browser = launch_headless_browser()
    
    # Setup WebGL context and shader
    setup_glsl_shader(browser, shader_code, params)
    
    # Evaluate all test cases
    results = map(test_cases) do tc
        evaluate_glsl_point(browser, tc.wi, tc.wo)
    end
    
    close(browser)
    
    return results
end
```

### Browser Automation

**Option 1: Playwright.jl (recommended)**
```julia
using Playwright

function launch_headless_browser()
    playwright = Playwright.install()
    browser = playwright.chromium.launch(headless=true)
    page = browser.new_page()
    
    # Load minimal HTML with WebGL
    page.goto("about:blank")
    page.set_content("""
    <html><body>
    <canvas id="gltest" width="1" height="1"></canvas>
    <script>
        // WebGL setup code
        const canvas = document.getElementById('gltest');
        const gl = canvas.getContext('webgl2');
        // ... shader compilation etc
    </script>
    </body></html>
    """)
    
    return page
end
```

**Option 2: Manual Three.js Script (simpler but manual)**
```julia
# Write test to standalone HTML file
# Run in browser, save results to JSON
# Load results in Julia for comparison
```

### Unit Test Organization

```julia
# test/runtests.jl

using Test
using BRDFPlatform

@testset "BRDFPlatform.jl" begin
    @testset "Plugin Registration" begin
        @test length(FORMAT_PLUGINS) >= 1
        @test haskey(FORMAT_PLUGINS, "merlformat")
    end
    
    @testset "Format Loading" begin
        @testset "MERL" begin
            data = load_brdf(MERLFormat, "test/data/gold.binary")
            @test data.status == :ok
            @test size(data.samples) == (3, 90, 90, 180)
        end
    end
    
    @testset "Model Validation" begin
        @testset "Lambertian" begin
            model = LambertianModel(RGB(0.5, 0.5, 0.5))
            err = validate_model(model)
            @test err < 1e-4
        end
        
        @testset "Cook-Torrance" begin
            model = CookTorranceModel(0.3, 0.04, RGB(0.8, 0.8, 0.8))
            err = validate_model(model)
            @test err < 1e-3  # Slightly higher tolerance for complex model
        end
    end
    
    @testset "Optimization" begin
        # Test on synthetic data with known parameters
        true_model = LambertianModel(RGB(0.6, 0.6, 0.6))
        synthetic_data = generate_synthetic_brdf(true_model, noise=0.01)
        
        fitted = fit_brdf(
            LevenbergMarquardt(max_iterations=50),
            LambertianModel,
            synthetic_data
        )
        
        @test norm(fitted.albedo - true_model.albedo) < 0.1
    end
end
```

### Performance Benchmarks

```julia
# test/benchmarks.jl

using BenchmarkTools

@benchmark load_brdf(MERLFormat, "test/data/gold.binary")

@benchmark fit_brdf(
    LevenbergMarquardt(max_iterations=100),
    CookTorranceModel,
    merl_data
)

@benchmark evaluate(cook_torrance_model, wi, wo)
```

### Continuous Testing Strategy

**Development:**
- Run unit tests on every commit
- Run validation tests before merging
- Run benchmarks weekly

**Pre-submission:**
- Full test suite on all MERL materials
- Performance profiling
- Memory leak detection

### Reference Comparison (Optional)

Compare against external tools:
```julia
# Compare fitted parameters to ALTA results (if available)
function compare_to_alta(material_name)
    our_result = fit_model(material_name)
    alta_result = load_alta_reference("alta_results/$(material_name).txt")
    
    @test our_result ≈ alta_result rtol=0.1
end
```

---

## 11. Development Workflow (Implementation Guide)

### Phase 1: Core Infrastructure (Week 1)
1. Create `types.jl` with abstract types
2. Implement `MeasuredBRDF` struct
3. Build plugin discovery system
4. Test with one dummy plugin of each type

### Phase 2: First Working Plugin (Week 1-2)
1. Integrate existing MERL loader
2. Implement Lambertian model (simplest)
3. Create fallback GLSL shader
4. Basic Bonito UI showing loaded data

### Phase 3: First Complete Pipeline (Week 2-3)
1. Add Levenberg-Marquardt optimizer
2. Implement Cook-Torrance model (Julia + GLSL)
3. Connect Three.js rendering
4. WGLMakie hemisphere plot

### Phase 4: Expand (Week 3-4)
1. Add BiRD format support
2. Add 2-3 more analytical models
3. Add error visualization
4. Model comparison UI

**After this:** Neural BRDF integration (Phase 3 of thesis timeline)

---

## Summary

**Key Design Principles:**
- ✅ Flat structure (no deep nesting)
- ✅ Abstract types for interchangeability
- ✅ Convention-based plugin system
- ✅ Auto-detection for formats
- ✅ Dual implementation (Julia + GLSL) for models
- ✅ Graceful fallbacks (table-based rendering)
- ✅ Hot reload support for development
- ✅ Minimal Julia-JS boundary crossings

**What's NOT in scope:**
- ❌ Complex inheritance hierarchies
- ❌ Macro-based DSLs
- ❌ Over-abstraction
- ❌ Per-frame Julia-JS communication
