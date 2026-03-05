# BRDF Analysis Platform - Implementation TODO

**Created:** January 2025  
**Status:** Pre-Implementation  
**Timeline:** 5 months (Jan-May 2025)

---

## Priority 1: Critical (Fix Before Implementation)

### Architecture Fixes
- [x] Fix plugin registration system (explicit, not auto-discovery)
- [x] Fix MeasuredBRDF structure (handle different sampling patterns)
- [x] Fix format auto-detection (magic bytes, proper thresholds)
- [x] Clarify neural BRDF is optional in timeline
- [x] Define Three.js integration details
- [ ] Add validation testing framework (Julia vs GLSL comparison)
- [ ] Define fallback chain error handling

### Documentation
- [ ] Create example plugin file for each type (format/model/optimizer)
- [ ] Write plugin developer guide
- [ ] Document format-specific data types and their lookup conventions
- [ ] Document BRDF evaluation conventions (coordinate systems, units)

---

## Phase 1: Foundation (Month 1) - Week 1-4

### Week 1: Core Infrastructure
- [ ] Create project structure
  ```
  BRDFPlatform.jl/
  â”œâ”€â”€ src/
  â”‚   â”œâ”€â”€ BRDFPlatform.jl (main module)
  â”‚   â”œâ”€â”€ types.jl (abstract types)
  â”‚   â”œâ”€â”€ formats/
  â”‚   â”œâ”€â”€ models/
  â”‚   â”œâ”€â”€ optimizers/
  â”‚   â”œâ”€â”€ rendering/
  â”‚   â”œâ”€â”€ visualization/
  â”‚   â””â”€â”€ app/
  â”œâ”€â”€ test/
  â””â”€â”€ examples/
  ```

- [ ] Implement types.jl
  - [ ] Abstract types (BRDFData, BRDFModel, BRDFOptimizer)
  - [x] ~~SamplingPattern hierarchy~~ (removed — sampling is internal to each format type)
  - [ ] MeasuredBRDF struct
  - [ ] Basic helper functions (coordinate conversions)

- [ ] Implement plugin registration system
  - [ ] Register functions (register_format!, register_model!, register_optimizer!)
  - [ ] Plugin registries (FORMAT_PLUGINS, MODEL_PLUGINS, OPTIMIZER_PLUGINS)
  - [ ] load_plugins! function
  - [ ] User plugin support (~/.brdf_platform/)

- [ ] Basic testing
  - [ ] Test plugin registration works
  - [ ] Test MeasuredBRDF construction
  - [ ] Test coordinate conversions

### Week 2: First Format Plugin (MERL)
- [ ] Research MERL format specification
  - [ ] Understand Rusinkiewicz parameterization
  - [ ] Document coordinate system
  - [ ] Find test files

- [ ] Implement formats/merl.jl
  - [ ] MERLFormat type
  - [ ] detect_format implementation (file size check)
  - [ ] load_brdf implementation (binary reading)
  - [ ] MERLData construction with grid dimensions
  - [ ] Register plugin

- [ ] Testing
  - [ ] Test MERL detection on real files
  - [ ] Test MERL loading on multiple materials
  - [ ] Verify data dimensions
  - [ ] Test auto-detection (load_brdf_auto)

- [ ] MERL â†’ Standard conversion utilities
  - [ ] Rusinkiewicz â†’ spherical coordinate conversion
  - [ ] Sampling point extraction

### Week 3: First Model Plugin (Lambertian)
- [ ] Implement models/lambertian.jl
  - [ ] LambertianModel type (single parameter: albedo)
  - [ ] evaluate function (Julia)
  - [ ] glsl_shader function
  - [ ] shader_parameters function
  - [ ] Register plugin

- [ ] Create GLSL shader
  - [ ] Write shaders/lambertian.glsl
  - [ ] Test in browser console
  - [ ] Verify matches Julia implementation

- [ ] Validation testing
  - [ ] Compare Julia vs GLSL on test inputs
  - [ ] Visual inspection (hemisphere plot)
  - [ ] Numerical comparison (max error)

### Week 4: Basic Bonito UI + Visualization
- [ ] Setup Bonito environment
  - [ ] Test basic App rendering
  - [ ] Test WGLMakie integration
  - [ ] Test Three.js embedding

- [ ] Implement visualization/plots.jl
  - [ ] Hemisphere plot (WGLMakie)
  - [ ] 2D slice plots
  - [ ] Color-coded visualization

- [ ] Implement app/main_app.jl
  - [ ] File upload widget
  - [ ] Format auto-detection display
  - [ ] Basic hemisphere plot
  - [ ] Three.js sphere renderer (basic material)

- [ ] **Milestone:** Can load MERL file and visualize it

---

## Phase 2: Analytical Models (Months 2-3) - Week 5-12

### Week 5: Three.js Renderer Framework
- [ ] Implement rendering/threejs_renderer.jl
  - [ ] create_threejs_renderer function
  - [ ] update_brdf_material function
  - [ ] update_brdf_uniforms function
  - [ ] Fallback error handling

- [ ] Test with Lambertian model
  - [ ] Verify shader compilation
  - [ ] Test uniform updates
  - [ ] Test parameter sliders

- [ ] Implement fallback renderer
  - [ ] Create shaders/fallback.glsl (lookup table)
  - [ ] Texture upload from MeasuredBRDF
  - [ ] Automatic fallback on shader error

### Week 6: Cook-Torrance Model
- [ ] Research Cook-Torrance BRDF
  - [ ] Understand microfacet theory
  - [ ] Parameter ranges and meanings
  - [ ] Find reference implementations

- [ ] Implement models/cook_torrance.jl
  - [ ] CookTorranceModel type (roughness, F0, albedo)
  - [ ] evaluate function (Julia)
  - [ ] Normal distribution function (GGX)
  - [ ] Fresnel term (Schlick approximation)
  - [ ] Geometry term (Smith)

- [ ] Create GLSL shader
  - [ ] Write shaders/cook_torrance.glsl
  - [ ] Port Julia math exactly
  - [ ] Test compilation

- [ ] Validation
  - [ ] Julia vs GLSL comparison tests
  - [ ] Compare to PBRT reference
  - [ ] Visual verification

### Week 7: Levenberg-Marquardt Optimizer
- [ ] Research LM algorithm
  - [ ] Understand Jacobian computation
  - [ ] Find Julia LM packages (LsqFit.jl)
  - [ ] Study convergence criteria

- [ ] Implement optimizers/levenberg_marquardt.jl
  - [ ] LevenbergMarquardt type (max_iter, tol)
  - [ ] fit_brdf function
  - [ ] Objective function (RMSE or RMS-LogE)
  - [ ] Jacobian computation (finite differences or automatic)
  - [ ] Parameter bounds enforcement

- [ ] Testing
  - [ ] Test on synthetic data (known parameters)
  - [ ] Test on MERL data
  - [ ] Verify convergence
  - [ ] Check parameter validity

### Week 8: Error Metrics & Visualization
- [ ] Implement error metrics
  - [ ] RMSE computation
  - [ ] RMS-LogE with Îµ=0.001
  - [ ] Maximum error
  - [ ] Per-region analysis

- [ ] Create error visualizations
  - [ ] WGLMakie heatmap (angular space)
  - [ ] Error histogram
  - [ ] Per-wavelength error

- [ ] Add to UI
  - [ ] Error metric cards
  - [ ] Interactive heatmap
  - [ ] Comparison table

### Week 9-10: Additional Analytical Models
- [ ] Blinn-Phong model
  - [ ] Implementation + GLSL
  - [ ] Validation
  - [ ] Integration

- [ ] Ward model
  - [ ] Implementation + GLSL  
  - [ ] Anisotropy support
  - [ ] Validation

- [ ] Model comparison UI
  - [ ] Side-by-side rendering
  - [ ] Parameter table
  - [ ] Error comparison

### Week 11: BiRD Format Support
- [ ] Research BiRD format
  - [ ] Understand JSON structure
  - [ ] Study irregular sampling
  - [ ] Find example files

- [ ] Implement formats/bird.jl
  - [ ] BiRDFormat type
  - [ ] detect_format (JSON structure check)
  - [ ] load_brdf (JSON parsing)
  - [ ] BiRDData construction with direction lists

- [ ] Testing
  - [ ] Test on BiRD files
  - [ ] Verify auto-detection
  - [ ] Compare to MERL

### Week 12: Polish & Integration
- [ ] User plugin system UI
  - [ ] "Reload Plugins" button
  - [ ] Plugin list display
  - [ ] Error reporting

- [ ] Documentation
  - [ ] User guide
  - [ ] Plugin development guide
  - [ ] API documentation

- [ ] **Milestone:** End-to-end analytical pipeline working

---

## Phase 3: Neural BRDF (Month 3) [OPTIONAL]

**Decision point:** Only start if Week 12 completed early.

### IF PROCEEDING:
- [ ] Literature review
  - [ ] Study DeepBRDF paper
  - [ ] Study NeuBRDF paper
  - [ ] Understand training requirements

- [ ] Choose architecture
  - [ ] DeepBRDF (material editing focus)
  - [ ] NeuBRDF (cutting-edge)
  - [ ] Hybrid approach?

- [ ] Implementation
  - [ ] models/neural_brdf.jl
  - [ ] Flux.jl/Lux.jl setup
  - [ ] Pre-trained weights loading
  - [ ] GLSL approximation (if feasible)

- [ ] Integration
  - [ ] Add to model comparison
  - [ ] Neural vs analytical analysis

### IF SKIPPING:
- [ ] Document decision in thesis
- [ ] Note as future work
- [ ] Proceed to Phase 4

---

## Phase 4: Error Visualization & Comparison (Months 3-4)

### Advanced Error Metrics
- [ ] Perceptual error metrics
- [ ] Angular region analysis
- [ ] Wavelength-dependent errors
- [ ] Statistical summaries

### Rendering-Based Comparison
- [ ] Pixel-wise difference visualization
- [ ] Image quality metrics (SSIM, etc.)
- [ ] Side-by-side material rendering
- [ ] Multiple lighting conditions

### Interactive Model Selection
- [ ] Automated model ranking
- [ ] Manual model exploration
- [ ] Parameter sensitivity analysis
- [ ] Export fitted models

### UI Polish
- [ ] Layout improvements
- [ ] Responsive design
- [ ] Performance optimization
- [ ] User feedback

---

## Phase 5: Format Study & Refinement (Month 4)

### Empirical Format Comparison
- [ ] Test suite of materials
  - [ ] Select diverse MERL materials
  - [ ] Convert to BiRD
  - [ ] Measure conversion quality

- [ ] Performance benchmarks
  - [ ] Loading time
  - [ ] Memory footprint
  - [ ] Processing overhead

- [ ] Expressiveness study
  - [ ] What can each format represent?
  - [ ] What is lost in conversion?
  - [ ] Edge cases and limitations

- [ ] Documentation
  - [ ] Comparison table
  - [ ] Graphs and charts
  - [ ] Recommendations

### Additional Features (if time)
- [ ] Nelder-Mead optimizer
- [ ] Ashikhmin-Shirley model
- [ ] UTIA format support
- [ ] Performance optimization

### Testing & Validation
- [ ] Systematic testing across MERL library
- [ ] Cross-validation with other tools
- [ ] User testing
- [ ] Bug fixes

---

## Phase 6: Documentation (Months 4-5)

### Thesis Writing
- [ ] Introduction (motivation, research questions)
- [ ] Foundations (BRDF theory, related work)
- [ ] Design (architecture, plugin system)
- [ ] Implementation (key components, challenges)
- [ ] Evaluation (results, format comparison)
- [ ] Discussion (insights, limitations)
- [ ] Conclusion (contributions, future work)

### Code Documentation
- [ ] API documentation (Documenter.jl)
- [ ] Plugin developer guide
- [ ] User manual
- [ ] Example gallery
- [ ] README

### Presentation
- [ ] Slide deck
- [ ] Demo preparation
- [ ] Video recording
- [ ] Code repository cleanup

### Final Deliverables
- [ ] Thesis PDF (~50-60 pages)
- [ ] Source code (GitHub)
- [ ] Documentation website
- [ ] Demo video
- [ ] Presentation slides

---

## Ongoing Tasks (Throughout)

### Testing
- [ ] Unit tests for each component
- [ ] Integration tests
- [ ] Validation against references
- [ ] Performance benchmarks

### Code Quality
- [ ] Follow Julia style guide
- [ ] Document all public APIs
- [ ] Refactor as needed
- [ ] Code review

### Git Hygiene
- [ ] Meaningful commit messages
- [ ] Feature branches
- [ ] Regular commits
- [ ] Clean history

### Communication
- [ ] Weekly supervisor meetings
- [ ] Progress documentation
- [ ] Issue tracking
- [ ] Decision log

---

## Risk Mitigation

### High Priority Risks
1. **Three.js integration fails**
   - Mitigation: Prototype in Week 5, fallback to static images
   
2. **GLSL/Julia divergence**
   - Mitigation: Automated validation tests, visual comparison tools
   
3. **Timeline slip**
   - Mitigation: Neural BRDF is optional, strict prioritization

4. **Format conversion quality issues**
   - Mitigation: Document limitations, provide clear warnings

### Medium Priority Risks
1. **Performance issues with large datasets**
   - Mitigation: Benchmark early, optimize critical paths
   
2. **User plugin system complexity**
   - Mitigation: Simplify if needed, good examples

3. **Browser compatibility**
   - Mitigation: Test on Chrome/Firefox, document requirements

---

## Success Metrics

### Technical
- [ ] Load and visualize MERL files
- [ ] Fit 2+ analytical models
- [ ] Render in Three.js with custom shaders
- [ ] User can add own plugins
- [ ] Format comparison data collected

### Academic
- [ ] Thesis written and submitted
- [ ] Research questions answered
- [ ] Novel contributions documented
- [ ] Limitations acknowledged

### Open Source
- [ ] Code published on GitHub
- [ ] Documentation comprehensive
- [ ] Examples provided
- [ ] Reusable by community

---

## Notes

- **Flexibility:** Adjust timeline as needed based on actual progress
- **Focus:** Core functionality > polish
- **Documentation:** Write as you go, not at the end
- **Testing:** Don't skip validation tests
- **Communication:** Ask for help early when stuck

**Last Updated:** January 2025
