# BRDF Visualization Quick Reference

## Available Visualization Strategies

### 1. Hemisphere Plot (3D Surface)

**Purpose**: Shows the full 3D reflection distribution on a hemisphere

**Creates**: Single 3D plot with colored surface

**Best for**:
- Quick visual overview
- Material appearance
- Identifying specular lobes
- Understanding overall reflection shape

**Usage**:
```julia
# Simple version (45° incident, defaults)
fig = create_simple_hemisphere_plot(data)

# Custom version
fig = create_hemisphere_plot(
    data,
    Float32(π/4),  # wi_theta: incident polar angle
    Float32(0.0),  # wi_phi: incident azimuthal angle
    resolution=50  # grid resolution (higher = smoother but slower)
)
```

**Output**: Single 3D surface plot colored by BRDF values

---

### 2. Polar Slice Plot (Angular Analysis)

**Purpose**: Shows BRDF variation with azimuthal angle for different elevation angles

**Creates**: 4 polar plots (Red, Green, Blue, Luminance)

**Best for**:
- Quantitative analysis
- Identifying anisotropy
- Comparing different viewing angles
- Scientific measurements

**Usage**:
```julia
# Simple version (45° incident, 5 standard slices)
fig = create_simple_polar_slice_plot(data)

# Custom version
fig = create_polar_slice_plot(
    data,
    Float32(π/4),  # wi_theta: incident polar angle
    Float32(0.0),  # wi_phi: incident azimuthal angle
    theta_out_values=Float32[deg2rad(15), deg2rad(30), deg2rad(60)],
    n_phi_samples=180  # azimuthal resolution
)
```

**Output**: 4 polar subplots + legend

---

## Quick Comparison Table

| Feature | Hemisphere Plot | Polar Slice Plot |
|---------|----------------|------------------|
| **Dimensions** | 3D surface | 2D polar curves |
| **Plots** | 1 plot | 4 plots (R, G, B, Lum) |
| **Main Variable** | Full ω_out | φ_out at fixed θ_out |
| **Speed** | Slower (N² evaluations) | Faster (N×M evaluations) |
| **Anisotropy** | Visible in shape | Clear in pattern |
| **Quantitative** | Harder to read values | Easy to read values |
| **File Size** | Larger | Smaller |
| **Default Resolution** | 50×50 = 2,500 samples | 5×180 = 900 samples |

---

## Coordinate System Reference

Both visualizations use the same coordinate system:

```
Surface Normal: z-axis (0, 0, 1)
Hemisphere: z ≥ 0

Incident Direction (ωi):
  - wi_theta: polar angle from normal (0 to π/2)
  - wi_phi: azimuthal angle (0 to 2π)

Outgoing Direction (ωo):
  - theta_out: polar angle from normal
  - phi_out: azimuthal angle
```

---

## Common Patterns to Look For

### Specular Materials (Chrome, Polished Metal)
- **Hemisphere**: Sharp peak when ω_out = ω_in (mirror reflection)
- **Polar**: High values at specific angles, near-circular (isotropic)

### Diffuse Materials (Matte Surfaces)
- **Hemisphere**: Broad, smooth surface
- **Polar**: Nearly circular patterns, similar values at all angles

### Anisotropic Materials (Brushed Metal, Fabric)
- **Hemisphere**: Elongated or asymmetric lobes
- **Polar**: Non-circular patterns with directional peaks

---

## Complete Example Workflow

```julia
using BRDFPlatform

# Load BRDF
data = load_brdf("chrome.binary")

# 1. Quick overview with hemisphere plot
fig1 = create_simple_hemisphere_plot(data)
save("overview.png", fig1)

# 2. Detailed analysis with polar slices
fig2 = create_simple_polar_slice_plot(data)
save("analysis.png", fig2)

# 3. Custom incident angle comparison
for wi_theta in [Float32(π/6), Float32(π/4), Float32(π/3)]
    fig = create_polar_slice_plot(data, wi_theta, 0.0f0)
    angle_deg = round(rad2deg(wi_theta))
    save("polar_$(angle_deg)deg.png", fig)
end
```

---

## Performance Tips

### For Faster Rendering:

**Hemisphere Plot**:
```julia
# Lower resolution for preview
fig = create_hemisphere_plot(data, wi_theta, wi_phi, resolution=30)

# Higher resolution for final output
fig = create_hemisphere_plot(data, wi_theta, wi_phi, resolution=100)
```

**Polar Plot**:
```julia
# Fewer theta slices
theta_values = Float32[deg2rad(30), deg2rad(60)]

# Lower azimuthal resolution
fig = create_polar_slice_plot(data, wi_theta, wi_phi,
                              theta_out_values=theta_values,
                              n_phi_samples=90)
```

---

## Troubleshooting

### Issue: Empty or zero plots
**Check**:
1. `data.status == :ok` (file loaded successfully)
2. Angles are in radians, not degrees
3. Material has non-zero reflectance

### Issue: Hemisphere plot is noisy
**Solution**: Increase resolution
```julia
fig = create_hemisphere_plot(data, wi_theta, wi_phi, resolution=100)
```

### Issue: Polar plot curves are jagged
**Solution**: Increase n_phi_samples
```julia
fig = create_polar_slice_plot(..., n_phi_samples=360)
```

### Issue: Can't see color differences
**Solution**: Try a colored material or focus on luminance channel
```julia
data = load_brdf("gold-paint.binary")  # Colored material
```

---

## API Summary

### Functions

```julia
# Hemisphere plots
create_hemisphere_plot(data, wi_theta, wi_phi; resolution=50)
create_simple_hemisphere_plot(data; wi_theta=π/4, wi_phi=0.0)

# Polar plots
create_polar_slice_plot(data, wi_theta, wi_phi;
                        theta_out_values=[...], n_phi_samples=180)
create_simple_polar_slice_plot(data; wi_theta=π/4, wi_phi=0.0)
```

### All angles in RADIANS!

```julia
# Convert degrees to radians
Float32(deg2rad(45))  # 45° → π/4

# Convert radians to degrees
rad2deg(π/4)  # π/4 → 45°
```

---

## Test Scripts

Three test scripts are available:

1. **`test_simple_lookup.jl`**: Basic loading and lookup test (no plotting)
2. **`test_plot_scratch.jl`**: Tests both hemisphere and polar plots
3. **`test_polar_plot.jl`**: Focused polar plot test

Run with:
```bash
julia --project=. test_plot_scratch.jl
```

---

## Further Reading

- **Polar Plot Guide**: `docs/polar_plot_guide.md`
- **Project Documentation**: `CLAUDE.md`
- **Architecture**: `Architecture_Design.md`
