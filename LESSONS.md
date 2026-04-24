# Lessons Learned

Short notes on non-obvious problems and their fixes. One entry per issue.

---

## LM fit collapses on shiny materials (e.g. gold-metallic-paint)

**Symptom:** Fit returns `roughness` pinned to lower bound, `F0 ≈ 0`, only a weak
diffuse albedo. Rendered result looks nothing like the measurement.

**Cause:** Two compounding issues:
1. Sampling `(wi, wo)` uniformly on the hemisphere almost never hits a sharp
   specular lobe — the optimizer sees ~flat data and fits a weak Lambertian.
2. Linear-space residuals are dominated by the few specular-peak samples,
   which are too noisy (too few) to pull the optimizer toward the true peak.

**Fix:** In `LevenbergMarquardt` (`src/optimizers/levenberg_marquardt.jl`):
- Sample in Rusinkiewicz `(θ_h, φ_h, θ_d, φ_d)` space with `θ_h` biased toward
  0 — puts dense samples *on* the specular lobe.
- Fit `log1p(f)` instead of raw `f` — compresses the peak/diffuse dynamic range.

Both are now defaults (`sampling=:rusinkiewicz`, `loss=:log`). For pure metals,
also prefer `GGXSpecularModel`/`CookTorranceSpecularModel` over the diffuse-plus-specular variants.
