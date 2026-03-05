# CLAUDE.MD - AI Assistant Guide

**Project:** BRDF Analysis Platform
**Type:** Bachelor Thesis - Computer Graphics
**Student:** Maria Alexandra Kley
**Timeline:** January - May 2025 (5 months)
**Last Updated:** January 30, 2025 (Added documentation bloat rule)

---

## Project Overview

This is a Bachelor thesis project building an interactive web-based platform for analyzing and fitting BRDF (Bidirectional Reflectance Distribution Function) models to measured material data. The platform supports multiple input formats, analytical BRDF models, optimization algorithms, and provides both scientific visualization and realistic material rendering.

### Core Purpose
- Load measured BRDF data from various formats (MERL binary, BiRD/JSON)
- Fit multiple analytical BRDF models (Lambertian, Cook-Torrance, Ward, etc.)
- Compare model quality using both mathematical metrics and visual rendering
- Support extensibility through a plugin system
- Provide educational and research tool for graphics community

---

## Technology Stack

**DO NOT CHANGE THE STACK - These are research constraints:**

### Primary Stack
- **Julia** - All computation, fitting, scientific visualization
- **Bonito.jl** - Web framework (NOT Genie.jl)
- **WGLMakie** - Scientific plots (hemisphere plots, error heatmaps)
- **Three.js** - Material rendering (embedded in Bonito, NOT standalone)
- **GLSL** - Custom shaders for BRDF models
- **Flux.jl or Lux.jl** - Neural BRDF models (optional Phase 3)

### What NOT to Use
- ❌ WASM for BRDF computation (adds complexity without benefit)
- ❌ Standalone JavaScript framework
- ❌ Genie.jl (use Bonito.jl)
- ❌ Other web frameworks

---

## Architecture Principles

### KISS Philosophy
- **Flat directory structure** - Everything visible at one level
- **Convention over configuration** - Standard interfaces, minimal boilerplate
- **Explicit plugin registration** - No magic, clear initialization
- **Graceful degradation** - Fallbacks for errors, never crash

### Plugin System
The platform is built around three plugin types:

1. **Format Plugins** (`src/formats/`)
   - Parse different BRDF file formats
   - Auto-detection via confidence scoring
   - Convert to standard `MeasuredBRDF` structure

2. **Model Plugins** (`src/models/`)
   - Dual implementation: Julia + GLSL
   - Julia: for fitting and error computation
   - GLSL: for real-time rendering in browser

3. **Optimizer Plugins** (`src/optimizers/`)
   - Fitting algorithms (Levenberg-Marquardt, Nelder-Mead, etc.)
   - Take model type + data → return fitted parameters

### Critical Design Decisions

**Dual Implementation Requirement:**
Every BRDF model MUST be implemented twice:
- Julia version: Used for optimization and error metrics
- GLSL version: Used for Three.js rendering

These MUST produce identical results - validation testing is mandatory.

**Data Flow:**
```
File Upload → Format Detection → MeasuredBRDF
    ↓
Model Selection → Fitting (Julia) → Parameters
    ↓
Visualization:
  - WGLMakie: Scientific plots (Julia)
  - Three.js: Material rendering (GLSL shaders)
```

**Sampling:**
Different formats use different sampling strategies internally:
- MERL: Rusinkiewicz half-angle parameterization (encoded in `MERLData`)
- BiRD: Irregular sampling with explicit directions

Sampling knowledge is encapsulated in each format's internal data type, not in a separate abstraction. All formats produce a `MeasuredBRDF` whose `_data` field handles lookup internally.

---

## Project Structure

```
BRDFToolkit/
├── src/
│   ├── BRDFPlatform.jl          # Main module, plugin discovery
│   ├── types.jl                  # Abstract types, MeasuredBRDF
│   │
│   ├── formats/                  # Input format plugins
│   │   ├── merl.jl              # MERL binary (.binary)
│   │   └── bird.jl              # BiRD format (.bird/.json)
│   │
│   ├── models/                   # BRDF model plugins
│   │   ├── lambertian.jl        # Diffuse baseline
│   │   ├── cook_torrance.jl     # Microfacet model
│   │   └── ward.jl              # Anisotropic model
│   │
│   ├── optimizers/               # Fitting algorithms
│   │   ├── levenberg_marquardt.jl
│   │   └── nelder_mead.jl
│   │
│   ├── rendering/                # Three.js integration
│   │   ├── threejs_renderer.jl  # Bonito ↔ Three.js bridge
│   │   └── shaders/             # GLSL implementations
│   │       ├── fallback.glsl    # Table-based fallback
│   │       ├── lambertian.glsl
│   │       └── cook_torrance.glsl
│   │
│   ├── visualization/            # Scientific visualization
│   │   └── plots.jl             # WGLMakie plots
│   │
│   └── app/                      # Web interface
│       └── main_app.jl          # Bonito UI composition
│
├── test/
│   ├── runtests.jl              # Test suite
│   ├── validation.jl            # Julia vs GLSL validation
│   └── benchmarks.jl            # Performance tests
│
├── examples/                     # Example plugins
│
├── PROJECT.md                    # Project requirements
├── Architecture_Design.md        # Detailed architecture
├── TODO.md                       # Implementation checklist
└── CLAUDE.MD                     # This file
```

---

## Key Interfaces

### Format Plugin Interface
```julia
# Required methods:
detect_format(::Type{<:BRDFData}, path::String)::Float64
load_brdf(::Type{<:BRDFData}, path::String)::MeasuredBRDF

# Registration:
register_format!(FormatType)
```

### Model Plugin Interface
```julia
# Required methods:
evaluate(model::BRDFModel, wi::Vec3, wo::Vec3)::RGB{Float32}
glsl_shader(model::BRDFModel)::Tuple{String, Vector{String}}
shader_parameters(model::BRDFModel)::Dict{String, Any}

# Registration:
register_model!(ModelType)
```

### Optimizer Plugin Interface
```julia
# Required methods:
fit_brdf(
    optimizer::BRDFOptimizer,
    model_type::Type{<:BRDFModel},
    data::MeasuredBRDF
)::BRDFModel

# Registration:
register_optimizer!(OptimizerType)
```

---

## Implementation Timeline

### Phase 1: Foundation (Month 1) ✓ DESIGN COMPLETE
- Core types and plugin system
- MERL format loader
- Lambertian model (simplest)
- Basic Bonito UI + WGLMakie visualization

### Phase 2: Analytical Models (Months 2-3) ← CURRENT FOCUS
- Three.js renderer framework
- Cook-Torrance model
- Levenberg-Marquardt optimizer
- Error metrics and visualization
- Additional models (Blinn-Phong, Ward)
- BiRD format support

### Phase 3: Neural BRDF (Month 3) - OPTIONAL
**ONLY proceed if Phase 2 completed early!**
- Flux.jl/Lux.jl setup
- DeepBRDF or NeuBRDF implementation
- Neural vs analytical comparison
- If skipped: Document as future work

### Phase 4: Error Visualization (Months 3-4)
- Advanced error metrics
- Rendering-based comparison
- Interactive model selection

### Phase 5: Format Study (Month 4)
- Empirical MERL vs BiRD comparison
- Performance benchmarks
- Additional features if time permits

### Phase 6: Documentation (Months 4-5)
- Thesis writing (~50-60 pages)
- Code documentation
- Presentation preparation

---

## Research Questions

1. **RQ1: Plugin System Extensibility**
   How can a plugin system provide genuine extensibility across all components while maintaining clean interfaces?

2. **RQ2: Error Visualization**
   What combination of BRDF-space and image-space error visualization best helps users select appropriate models?

3. **RQ3: Interactive Model Comparison**
   How does interactive comparison support quality assessment versus automated selection?

4. **RQ4: Universal Format Feasibility**
   Is there a "best" BRDF data format? What are empirical trade-offs between universal (.bird) and specialized (MERL) formats?

---

## Development Guidelines

### When Writing Code

**DO:**
- ✅ Follow flat directory structure
- ✅ Use explicit plugin registration
- ✅ Implement both Julia and GLSL versions for models
- ✅ Write validation tests comparing Julia vs GLSL
- ✅ Handle errors gracefully with fallbacks
- ✅ Document coordinate systems and units
- ✅ Use type system for interchangeability

**DON'T:**
- ❌ Create deep module hierarchies
- ❌ Use macro-based DSLs for plugins
- ❌ Skip GLSL validation tests
- ❌ Make per-frame Julia-JS calls (performance!)
- ❌ Change the core technology stack
- ❌ Create additional documents when fixing problems (fix only what's broken)

### Testing Requirements

**Critical:**
- Julia vs GLSL validation for EVERY model
- Maximum relative error < 1e-4 (Lambertian) or < 1e-3 (complex models)
- Visual comparison in addition to numerical tests

**Suggested approach:**
- Use headless browser (Playwright.jl) for automated GLSL testing
- Generate random test directions
- Compare results across 1000+ samples
- Report worst-case scenarios

### Performance Considerations

**Expensive operations (Julia):**
- BRDF fitting optimization
- Error metric computation
- Format loading/conversion

**Cheap operations (Browser):**
- Shader compilation (once per model)
- Uniform updates (real-time)
- Material rendering (60 FPS)

**Optimization strategy:**
- Minimize Julia → JavaScript communication
- Send GLSL code once, update uniforms only
- No communication during rotation/animation

---

## Common Tasks & Patterns

### Adding a New Format Plugin

1. Create `src/formats/your_format.jl`
2. Implement `YourFormat <: BRDFData`
3. Create an internal data struct (e.g. `YourData`) that encapsulates sampling knowledge
4. Implement `detect_format` (return confidence 0.0-1.0)
5. Implement `load_brdf` (return `MeasuredBRDF` wrapping your data struct)
6. Implement `_brdf_value(data::YourData, wavelengths, wi, wo)` for lookup
7. Implement `n_samples_str(data::YourData)` for display
8. Call `register_format!(YourFormat)` at end of file
9. Write tests in `test/formats/test_your_format.jl`

### Adding a New BRDF Model

1. Create `src/models/your_model.jl`
2. Implement `YourModel <: BRDFModel` struct with parameters
3. Implement `evaluate` function (Julia implementation)
4. Create `src/rendering/shaders/your_model.glsl`
5. Implement `glsl_shader` (return shader code + uniform names)
6. Implement `shader_parameters` (extract values from struct)
7. Call `register_model!(YourModel)` at end of file
8. Write validation test comparing Julia vs GLSL
9. Test with interactive UI

### Adding an Optimizer

1. Create `src/optimizers/your_optimizer.jl`
2. Implement `YourOptimizer <: BRDFOptimizer`
3. Implement `fit_brdf` for each supported model type
4. Handle parameter bounds and constraints
5. Call `register_optimizer!(YourOptimizer)` at end of file
6. Test on synthetic data with known parameters

---

## Error Handling Philosophy

**Never crash. Always degrade gracefully.**

### Fallback Chain

**Format Loading:**
```
Try all format plugins → Pick highest confidence (>0.6)
  ↓ Failed
Return MeasuredBRDF with status=:failed + error message
```

**Model Fitting:**
```
Try primary optimizer → Works? Done.
  ↓ Failed
Try fallback optimizer → Works? Use it (with warning)
  ↓ Failed
Return default parameters + error status
```

**Rendering:**
```
Try custom GLSL shader → Compiles? Use it.
  ↓ Failed
Use table-based fallback shader → Always works
```

### User Feedback
- Always display error messages in UI (not just console)
- Use color coding (red for errors, yellow for warnings)
- Provide actionable suggestions ("Try different optimizer")
- Never silently fail

---

## File Format Details

### MERL Binary Format
- Fixed size: 17,496,000 bytes
- Dimensions: 90 × 90 × 180 samples × 3 channels
- Parameterization: Rusinkiewicz half-angle
- Channels: RGB (approximated wavelengths: 630nm, 532nm, 465nm)
- Isotropic materials only (phi_h = 0)

### BiRD Format
- JSON-based structure
- Supports irregular sampling
- Explicit direction vectors per sample
- Can represent anisotropic materials
- More flexible but larger file size

### Universal Representation
All formats convert to `MeasuredBRDF`:
```julia
struct MeasuredBRDF <: BRDFData
    _data::Any                       # Internal storage (format-specific, opaque)
    wavelengths::Vector{Float32}     # nm
    name::String
    source_format::Symbol            # :merl, :bird, :utia
    status::Symbol                   # :ok, :partial, :failed
    error_msg::Union{Nothing, String}
end
```

---

## Important Coordinate Systems

**Standard conventions:**
- All vectors normalized
- Hemisphere: z-axis is surface normal
- Angles in radians
- wi: incident direction (pointing AWAY from surface)
- wo: outgoing direction (pointing AWAY from surface)
- All BRDF evaluations include cosine term handling

**Rusinkiewicz Parameterization (MERL):**
- theta_h: Half-angle polar (0 to π/2)
- phi_h: Half-angle azimuthal (0 for isotropic)
- theta_d: Difference polar (0 to π/2)
- phi_d: Difference azimuthal (0 to 2π)

---

## Success Criteria

### Minimum Viable Product (Must-Have)
- ✅ MERL binary format support
- ✅ BiRD format support
- ✅ 2-3 analytical models (Lambertian + Cook-Torrance minimum)
- ✅ 1-2 optimization algorithms (Levenberg-Marquardt minimum)
- ✅ WGLMakie visualization + Three.js rendering
- ✅ Basic error metrics
- ✅ Plugin system with user plugin support

### Enhanced Features (Should-Have)
- 5+ analytical models
- Format comparison study results
- Comprehensive error visualization
- Advanced rendering comparisons

### Aspirational (Nice-to-Have)
- Neural BRDF model integration
- UTIA format support
- Additional optimization algorithms
- Performance optimization

---

## Context for AI Assistants

### When I Ask For Help

**Prioritize:**
1. Correctness over performance (thesis project, not production)
2. Clarity over cleverness (code will be evaluated)
3. Research goals over engineering perfection
4. Working features over perfect architecture

**Remember:**
- 5-month timeline is tight - pragmatic solutions favored
- Neural BRDF is optional/exploratory
- Thesis focus: balance implementation + scientific analysis
- Plugin system is core to RQ1 - keep it clean and simple

**Ask clarifying questions about:**
- Which phase of implementation we're in
- Whether feature is must-have, should-have, or nice-to-have
- Trade-offs between approaches
- Testing requirements for specific components

### Common Pitfalls to Avoid

1. **Over-engineering** - Don't create abstractions for future flexibility beyond thesis scope
2. **Premature optimization** - Correctness first, performance only if needed
3. **Technology drift** - Don't suggest alternatives to the fixed stack
4. **Skipping validation** - Julia vs GLSL comparison is mandatory
5. **Monolithic commits** - Keep changes focused and testable
6. **Documentation bloat** - When fixing a problem, fix ONLY the problem. Do not create additional markdown files, summaries, or documentation unless explicitly requested

---

## Quick Reference Commands

### Development Setup
```bash
# Start Julia REPL in project
julia --project=.

# Activate environment
] activate .

# Install dependencies
] instantiate

# Run tests
] test

# Start Bonito app
julia> using BRDFPlatform
julia> run_app()
```

### Testing
```bash
# Run all tests
julia --project=. test/runtests.jl

# Run validation tests only
julia --project=. test/validation.jl

# Run benchmarks
julia --project=. test/benchmarks.jl
```

### Plugin Development
```bash
# User plugin locations:
~/.brdf_platform/formats/
~/.brdf_platform/models/
~/.brdf_platform/optimizers/

# Reload plugins in UI: Click "🔄 Reload User Plugins" button
```

---

## Resources

### Key Papers
- Rusinkiewicz MERL BRDF Database
- BiRD format specification
- Cook-Torrance microfacet model
- DeepBRDF (if pursuing neural models)

### External Tools (for comparison)
- ALTA (BRDF fitting library)
- PBRT (reference renderer)
- MERL BRDF Database

### Julia Packages
- Bonito.jl - Web framework
- WGLMakie.jl - WebGL visualization
- Flux.jl / Lux.jl - Neural networks (optional)
- LsqFit.jl - Levenberg-Marquardt
- Optim.jl - Optimization algorithms

---

## Notes for Future Work

If this project is extended beyond the thesis:
- Add spectral BRDF support (beyond RGB)
- Real-time BRDF editing
- Material appearance design tools
- BRDF database with search
- Cloud rendering service
- Mobile/tablet support
- VR/AR material preview

---

## Contact & Documentation

- **Related Files:** PROJECT.md, Architecture_Design.md, TODO.md
- **Test Data:** MERL database, BiRD examples
- **Issue Tracking:** (Add GitHub repo when created)

**This file should be updated as the project evolves.**

Last substantial update: January 30, 2025
