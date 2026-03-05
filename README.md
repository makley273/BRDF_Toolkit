# BRDF Platform

A Julia-based platform for analyzing and fitting BRDF (Bidirectional Reflectance Distribution Function) models to measured material data.

**Bachelor Thesis Project** - Maria Alexandra Kley
**Timeline:** January - May 2025

## Project Status

✅ **Phase 1 Foundation - Complete**
- Core type system and abstractions
- Plugin system with automatic discovery
- MERL format loader
- Basic test suite

✅ **Phase 1 + Visualization - Complete**
- Interactive hemisphere visualization
- WGLMakie 3D plotting
- Real-time BRDF evaluation
- File loading (path-based)
- Angle controls with live updates

🚧 **Phase 2 - Next Steps**
- Browser-based file upload (currently path-based only)
- Analytical BRDF models (Lambertian, Cook-Torrance, etc.)
- Three.js renderer integration
- Optimization algorithms

## Quick Start

### Installation

```bash
cd BRDFToolkit
julia --project=.
```

In Julia REPL:
```julia
] instantiate  # Install dependencies
```

### Testing

```bash
julia --project=. test/runtests.jl
```

### Starting the Web UI

**Option 1: Basic UI (Plugin registry view)**
```bash
julia --project=. start_app.jl
```

**Option 2: Interactive UI with Visualization (Recommended)**
```bash
julia --project=. start_interactive.jl
```

Then open your browser to: `http://127.0.0.1:8000`

**Features:**
- 📂 Load MERL .binary files (via file path input)
- 📊 Interactive hemisphere visualization
- 🎚️ Real-time incident angle controls
- 🌈 Color-coded reflection distribution
- 🎯 3D rotation and zoom

**How to Load a File:**
1. Enter the full path to your `.binary` file in the text field
   - Example: `C:/path/to/chrome.binary`
2. Click "Load File"
3. Adjust angle sliders to explore the BRDF!

> ⚠️ **Note:** Browser file upload widget is present but not yet fully functional (returns blob URLs). Use direct file path for now. See `FILE_UPLOAD_FIX.md` for details.

See `VISUALIZATION_GUIDE.md` for detailed visualization documentation.

### Basic Usage (Programmatic)

```julia
using BRDFPlatform

# List available plugins
plugins = list_plugins()

# Load BRDF data (auto-detect format)
data = load_brdf("path/to/material.binary")

# Check data status
println("Material: ", data.name)
println("Status: ", data.status)
println("Wavelengths: ", data.wavelengths)
```

## Project Structure

```
BRDFToolkit/
├── src/
│   ├── BRDFPlatform.jl      # Main module
│   ├── types.jl              # Core types and abstractions
│   ├── formats/              # Format loader plugins
│   │   └── merl.jl          # ✅ MERL binary format
│   ├── models/               # BRDF model plugins
│   ├── optimizers/           # Fitting algorithm plugins
│   ├── rendering/            # Three.js integration
│   │   └── shaders/         # GLSL shader implementations
│   ├── visualization/        # WGLMakie scientific plots
│   └── app/                  # Bonito web UI
│       └── main_app.jl      # ✅ Web interface
├── test/
│   └── runtests.jl          # Test suite
├── examples/                 # Example plugins
├── Project.toml              # Julia package definition
├── CLAUDE.MD                 # AI assistant guide
├── Architecture_Design.md    # Detailed architecture
├── TODO.md                   # Implementation checklist
└── README.md                 # This file
```

## Core Concepts

### Plugin System

The platform is built around three plugin types:

1. **Format Plugins** - Parse different BRDF file formats
   - Implement `detect_format()` and `load_brdf()`
   - Auto-registration via `register_format!()`

2. **Model Plugins** - Analytical BRDF models
   - Dual implementation: Julia + GLSL
   - Implement `evaluate()`, `glsl_shader()`, `shader_parameters()`

3. **Optimizer Plugins** - Fitting algorithms
   - Implement `fit_brdf(optimizer, model_type, data)`
   - Support different optimization strategies

### Universal Data Representation

All formats convert to `MeasuredBRDF`:
```julia
struct MeasuredBRDF <: BRDFData
    _data::Any                       # Internal storage (format-specific, opaque)
    wavelengths::Vector{Float32}     # Wavelengths in nm
    name::String
    source_format::Symbol
    status::Symbol                   # :ok, :partial, :failed
    error_msg::Union{Nothing, String}
end
```

## Supported Formats

- ✅ **MERL Binary** (.binary) - 90×90×180 Rusinkiewicz parameterization
- 🚧 **BiRD** (.bird/.json) - Irregular sampling, anisotropic support
- 📋 **UTIA** - Planned

## Development

### Adding a Format Plugin

1. Create `src/formats/your_format.jl`
2. Define format type: `struct YourFormat <: BRDFData end`
3. Implement `detect_format(::Type{YourFormat}, path)::Float64`
4. Implement `load_brdf(::Type{YourFormat}, path)::MeasuredBRDF`
5. Register: `register_format!(YourFormat)` at end of file

### Running Tests

```bash
# All tests
julia --project=. test/runtests.jl

# With verbose output
julia --project=. -e 'using Test; include("test/runtests.jl")'
```

### User Plugins

Place custom plugins in:
```
~/.brdf_platform/
├── formats/
├── models/
└── optimizers/
```

Reload with:
```julia
BRDFPlatform.load_user_plugins()
```

## Technology Stack

- **Julia 1.9+** - Core language
- **Bonito.jl** - Web framework
- **WGLMakie** - Scientific visualization
- **Three.js** - Material rendering
- **GLSL** - Custom shaders
- **LsqFit.jl** - Levenberg-Marquardt optimization
- **Optim.jl** - Additional optimizers

## Research Questions

This thesis investigates:
1. Plugin system design for BRDF analysis tools
2. Effective error visualization for model comparison
3. Interactive vs. automated model selection
4. Trade-offs between universal and specialized BRDF formats

## Documentation

- `CLAUDE.MD` - Comprehensive guide for AI assistants and developers
- `Architecture_Design.md` - Detailed system architecture
- `PROJECT.md` - Project requirements and specifications
- `TODO.md` - Implementation checklist and progress tracking

## Current Status Summary

**Completed:**
- ✅ Julia project structure
- ✅ Core type system (BRDFData, BRDFModel, BRDFOptimizer)
- ✅ Plugin registry and discovery
- ✅ MERL format loader with Rusinkiewicz parameterization
- ✅ Web UI with Bonito.jl
- ✅ **WGLMakie hemisphere visualization** ⭐ NEW
- ✅ **Interactive file upload with real-time plots** ⭐ NEW
- ✅ **Incident angle controls** ⭐ NEW
- ✅ Plugin registry display
- ✅ Test suite with 30 passing tests
- ✅ User plugin directory support

**Next Steps:**
- [ ] Implement Lambertian model (simplest BRDF)
- [ ] Create Three.js renderer integration
- [ ] Build basic Bonito UI
- [ ] Add WGLMakie visualization
- [ ] Implement Levenberg-Marquardt optimizer

## License

This is a Bachelor thesis project. Code will be made available under appropriate license upon completion.

## Author

Maria Alexandra Kley
Bachelor Thesis in Computer Graphics
January - May 2025
