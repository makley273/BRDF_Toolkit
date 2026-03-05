# Polar Slice Plot Guide

## Overview

The polar slice plot is a visualization technique for analyzing BRDF angular behavior. It shows how the BRDF varies with the **outgoing azimuthal angle (φ_out)** for different **outgoing polar angles (θ_out)**, given a fixed incident direction.

This visualization is particularly useful for:
- **Identifying anisotropic features**: Materials with direction-dependent reflectance
- **Comparing angular behavior**: How reflection changes with viewing elevation
- **Understanding material properties**: Specular lobes, diffuse components, retroreflection

## Visualization Structure

The polar slice plot creates **4 polar subplots**:

1. **Red Channel**: BRDF red component as a function of φ_out
2. **Green Channel**: BRDF green component as a function of φ_out
3. **Blue Channel**: BRDF blue component as a function of φ_out
4. **Luminance**: Overall brightness (weighted RGB combination)

Each subplot shows multiple curves, one for each outgoing polar angle (θ_out).

## Coordinate System

- **Incident Direction (ωi)**:
  - `wi_theta`: Polar angle from surface normal (0 to π/2)
  - `wi_phi`: Azimuthal angle (0 to 2π)
  - Fixed for the entire plot

- **Outgoing Direction (ωo)**:
  - `theta_out`: Polar angle from surface normal (fixed per curve)
  - `phi_out`: Azimuthal angle (varies around the circle, 0 to 2π)

## Usage

### Basic Usage

```julia
using BRDFPlatform

# Load BRDF data
data = load_brdf("path/to/material.binary")

# Create polar slice plot with default settings
fig = create_simple_polar_slice_plot(data)

# Display or save
display(fig)
save("polar_plot.png", fig)
```

### Advanced Usage

```julia
# Specify custom incident direction
wi_theta = Float32(π/3)  # 60° incident angle
wi_phi = Float32(0.0)    # Azimuth = 0°

# Specify which outgoing polar angles to plot
theta_out_values = Float32[
    deg2rad(10),   # Near-normal viewing
    deg2rad(30),   # Medium angle
    deg2rad(60),   # Grazing angle
    deg2rad(80)    # Very grazing
]

# Create plot with high resolution
fig = create_polar_slice_plot(
    data,
    wi_theta,
    wi_phi,
    theta_out_values=theta_out_values,
    n_phi_samples=360  # More samples for smoother curves
)
```

## Interpreting the Plots

### Isotropic Materials

For materials with **no directional dependence** (like ideal diffuse or smooth specular):
- All polar plots show **circular patterns** (constant at all φ_out angles)
- Example: Teflon, matte plastics

### Anisotropic Materials

For materials with **directional features** (brushed metals, fabrics):
- Polar plots show **non-circular patterns** (peaks and valleys)
- Peaks indicate preferred reflection directions
- Example: Brushed aluminum, wood grain

### Specular Materials

For materials with strong **specular reflection** (metals, glossy surfaces):
- High BRDF values when θ_out matches incident angle (specular lobe)
- Sharp peaks in the polar plots
- Lower values at other angles
- Example: Chrome, polished metals

### Diffuse Materials

For materials with **diffuse reflection** (matte, rough surfaces):
- Similar BRDF values at all angles (within hemisphere)
- Nearly circular polar patterns
- Example: Paper, unpolished surfaces

## Color Coding

Each θ_out value is plotted with a different color:
- Red, Blue, Green, Orange, Purple, Brown, Pink, Gray (cycles if more than 8)

The legend shows which color corresponds to which angle.

## Default Parameters

- **Default incident angle**: 45° (π/4 radians)
- **Default incident azimuth**: 0°
- **Default θ_out values**: [15°, 30°, 45°, 60°, 75°]
- **Default φ_out samples**: 180 (every 2 degrees)

## Examples

### Example 1: Chrome (Specular Metal)

```julia
data = load_brdf("chrome.binary")
fig = create_simple_polar_slice_plot(data)
```

**Expected result**:
- Strong peaks when θ_out = θ_in (specular reflection)
- Nearly circular (isotropic)
- High values in luminance plot

### Example 2: Comparing Different Incident Angles

```julia
data = load_brdf("brass.binary")

# Normal incidence
fig1 = create_polar_slice_plot(data, Float32(0.1), 0.0f0)

# Grazing incidence
fig2 = create_polar_slice_plot(data, Float32(π/3), 0.0f0)
```

**Expected result**:
- Different BRDF patterns for different incident angles
- Fresnel effects more visible at grazing angles

### Example 3: Anisotropic Material

```julia
# Load an anisotropic material (if available)
data = load_brdf("brushed_metal.binary")

fig = create_polar_slice_plot(
    data,
    Float32(π/4),
    0.0f0,
    theta_out_values=Float32[deg2rad(45)],  # Single angle
    n_phi_samples=360
)
```

**Expected result**:
- Non-circular pattern showing directional preference
- Peaks along brush direction

## Comparison with Hemisphere Plot

| Feature | Hemisphere Plot | Polar Slice Plot |
|---------|----------------|------------------|
| **Shows** | Full 3D reflection distribution | Angular slices at specific elevations |
| **Best for** | Overall material appearance | Detailed angular analysis |
| **Anisotropy** | Visible in 3D shape | Clear in polar pattern |
| **Quantitative** | Less precise | More precise (actual values visible) |
| **Use case** | Quick overview | Scientific analysis |

## API Reference

### `create_polar_slice_plot`

```julia
create_polar_slice_plot(
    data::MeasuredBRDF,
    wi_theta::Float32,
    wi_phi::Float32;
    theta_out_values::Vector{Float32} = Float32[π/6, π/4, π/3],
    n_phi_samples::Int = 180
) -> Figure
```

**Parameters:**
- `data`: Loaded BRDF data
- `wi_theta`: Incident polar angle (radians, 0 to π/2)
- `wi_phi`: Incident azimuthal angle (radians, 0 to 2π)
- `theta_out_values`: Outgoing polar angles to plot
- `n_phi_samples`: Number of azimuthal samples

**Returns:** WGLMakie Figure with 4 polar subplots

### `create_simple_polar_slice_plot`

```julia
create_simple_polar_slice_plot(
    data::MeasuredBRDF;
    wi_theta::Float32 = Float32(π/4),
    wi_phi::Float32 = 0.0f0
) -> Figure
```

Convenience wrapper with default parameters.

## Performance Notes

- **Evaluation count**: `n_theta_slices × n_phi_samples` BRDF lookups
- **Default cost**: 5 slices × 180 samples = 900 evaluations
- **Typical runtime**: < 1 second for MERL format
- For high-resolution plots (n_phi_samples=360), expect ~2 seconds

## Troubleshooting

### Issue: Plots are empty or all zeros

**Solution**: Check that:
1. BRDF data loaded successfully (`data.status == :ok`)
2. Incident angles are valid (θ ∈ [0, π/2], φ ∈ [0, 2π])
3. Material has non-zero reflectance

### Issue: Curves are noisy or jagged

**Solution**: Increase `n_phi_samples`:
```julia
fig = create_polar_slice_plot(data, wi_theta, wi_phi, n_phi_samples=360)
```

### Issue: Can't see differences between channels

**Solution**: Material might be neutral (gray). Try:
- A colored material (gold-paint, red-plastic)
- Check the luminance plot instead

## References

- MERL BRDF Database: http://www.merl.com/brdf/
- Rusinkiewicz parameterization for efficient BRDF sampling
- Polar coordinate visualization in computer graphics
