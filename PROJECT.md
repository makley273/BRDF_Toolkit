# BRDF Analysis Platform - Project Instructions

**Type:** Bachelor Thesis  
**Student:** Maria Alexandra Kley  
**Duration:** 5 months (January â€“ May 2025)  
**Last Updated:** January 2025

---

## Core Architecture Decision

**Technology Stack: Julia + Bonito + Embedded Three.js**

### Rationale
- **Neural BRDF requirement** makes Julia/Flux.jl essential (not optional)
- **Custom GLSL shaders** required for material rendering (cannot be replaced by WASM)
- **Hybrid approach** leverages strengths of both ecosystems

### Component Responsibilities

**Julia/Bonito Layer:**
- All computation (analytical fitting, neural BRDF inference, optimization)
- ML environment (Flux.jl or Lux.jl for neural models)
- Scientific visualization (WGLMakie for hemisphere plots, error heatmaps, polar diagrams)
- Plugin architecture and orchestration
- Data pipeline and format conversion

**Three.js Layer (Embedded):**
- Material appearance rendering with custom GLSL shaders
- Side-by-side visual comparison of fitted models
- Receives BRDF parameters from Julia via JavaScript interop
- Implements BRDF models in GLSL for real-time rendering

**Integration:**
```
Julia (Bonito) â†’ Computation â†’ WGLMakie scientific plots
                â†“
                Parameters â†’ Three.js canvas â†’ GLSL rendering
```

---

## Research Questions

### RQ1: Plugin System Extensibility
How can a plugin system provide genuine extensibility across all components (input formats, analytical models, neural models, optimization algorithms) while maintaining clean interfaces?

### RQ2: Error Visualization
What combination of BRDF-space and image-space error visualization best helps users select appropriate models for specific materials?

### RQ3: Interactive Model Comparison
How does interactive model comparison (including neural vs. analytical models) support quality assessment compared to automated selection alone?

### RQ4: Universal Format Feasibility
Is there a "best" BRDF data format, or is it even possible to define a format powerful enough to make others obsolete? What are empirical trade-offs between universal formats (.bird) and specialized formats (MERL binary)?

---

## Plugin Architecture

### Input Format Plugins
- MERL binary parser (.binary)
- Universal BiRD format reader (.bird/.json)
- UTIA format (if time permits)
- **Interface:** Standardized data structure â†’ internal representation

### BRDF Model Plugins

**Analytical Models (Julia):**
- Lambertian (diffuse baseline)
- Blinn-Phong
- Cook-Torrance
- Ward
- Ashikhmin-Shirley
- **Interface:** Parameter vector â†’ BRDF evaluation function

**Neural Models (Julia/Flux):**
- At least one neural BRDF architecture
- Pre-trained weights or training pipeline
- **Interface:** Input â†’ latent representation â†’ BRDF evaluation

**GLSL Shader Counterparts:**
- Each analytical model implemented in GLSL for rendering
- Neural model approximation or shader-compatible version

### Optimization Plugins
- Levenberg-Marquardt
- Nelder-Mead
- Adam (for neural model fine-tuning)
- **Interface:** Loss function + initial parameters â†’ optimized parameters

---

## Key Features

### 1. Interactive Model Comparison
- Automated loop through n models (analytical + neural)
- Side-by-side comparison in both BRDF-space and rendering-space
- Quality ranking with multiple metrics
- Manual model selection for exploration

### 2. Hybrid Error Visualization

**BRDF-Space Metrics:**
- RMSE (Root Mean Square Error)
- RMS-LogE with Îµ=0.001
- Maximum error with interactive heatmaps
- Per-region error analysis

**Rendering-Based Comparison:**
- Side-by-side material appearance (Three.js)
- Pixel-wise difference visualization
- Visual quality assessment

### 3. Neural BRDF Integration
- Flux.jl/Lux.jl for model training/inference
- Comparison against analytical models
- Exploration of neural vs. parametric trade-offs

---

## Implementation Phases

### Phase 1: Foundation (Month 1)
- Literature review (BRDF models, neural representations, optimization)
- Bonito + WGLMakie development environment setup
- Basic BRDF visualization (hemisphere plots)
- MERL format loader
- **Milestone:** Basic visualization pipeline functional

### Phase 2: Analytical Models (Months 2-3)
- Implement 3-5 analytical BRDF models in Julia
- Corresponding GLSL shaders for Three.js rendering
- Levenberg-Marquardt optimization integration
- Interactive parameter adjustment
- **Milestone:** End-to-end analytical fitting pipeline

### Phase 3: Neural BRDF (Month 3) [OPTIONAL - Nice-to-Have]
- **Only proceed if Phase 2 completed ahead of schedule**
- Flux.jl/Lux.jl environment setup
- Implement/integrate one neural BRDF architecture
- Training pipeline (if training from scratch) or weight loading
- Comparison framework (neural vs. analytical)
- **Milestone:** Neural model integrated with comparison UI

**If skipped:** Proceed directly to Phase 4 with analytical models only.

### Phase 4: Error Visualization & Comparison (Months 3-4)
- BRDF-space error metrics and heatmaps
- Three.js rendering integration
- Interactive model comparison interface
- Quality ranking system
- **Milestone:** Complete analysis workflow

### Phase 5: Format Study & Refinement (Month 4)
- Universal .bird format implementation
- Empirical format comparison (expressiveness, performance, conversion quality)
- Additional optimization algorithms
- Systematic testing across MERL materials
- **Milestone:** Evaluation-ready system

### Phase 6: Documentation (Months 4-5)
- Thesis writing (~50-60 pages)
- Code documentation and deployment guide
- Results analysis and visualization
- Presentation preparation
- **Milestone:** Thesis submission

---

## Technical Constraints

### Language Boundaries
- **Do NOT use WASM** for core BRDF logic (adds complexity without benefit)
- Julia handles all computation
- JavaScript/Three.js only for rendering visualization
- GLSL for shader implementations

### Framework Commitments
- **Bonito.jl** for web interface (not Genie.jl)
- **WGLMakie** for scientific plots
- **Three.js** embedded for material rendering (not standalone framework)
- **Flux.jl or Lux.jl** for neural models

### Plugin Interface Design
- Clean separation between plugin types
- Standardized data exchange formats
- Hot-swappable components without core modification
- Language-agnostic where possible (JSON for parameter passing)

---

## Expected Contributions

1. **First implementation** supporting universal .bird format with automated fitting
2. **Hybrid architecture** combining analytical and neural BRDF models
3. **Interactive model selection** interface with comparative analysis
4. **Empirical format comparison** study (addresses RQ4)
5. **Dual visualization approach** (BRDF-space + rendering-space)
6. **Open-source platform** for research and education

---

## Success Criteria

### Minimum Viable Product (Must-Have)
- Support for MERL binary and .bird formats
- 2-3 analytical models with fitting (Lambertian + Cook-Torrance minimum)
- 1-2 optimization algorithms (Levenberg-Marquardt minimum)
- Interactive visualization (WGLMakie + Three.js)
- Basic error metrics and comparison
- Plugin system with user plugin support

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

## Development Principles

1. **Modularity First:** Clean plugin interfaces over monolithic code
2. **Science Over Engineering:** Prioritize BRDF accuracy and comparison quality
3. **Pragmatic Choices:** 5-month timeline favors working solutions over perfect architecture
4. **Documentation Driven:** Code clarity for thesis evaluation
5. **Open Source Ready:** Reusable by graphics research community

---

## Notes

- **Timeline is tight:** Focus on core features first, extend only if time permits
- **Neural BRDF is exploratory:** May reveal insights about analytical model limitations
- **Format study is empirical:** Real measurements, not just theoretical comparison
- **Thesis focus:** Balance between implementation and scientific analysis
