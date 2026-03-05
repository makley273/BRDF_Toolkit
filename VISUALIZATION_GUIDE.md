## BRDF Visualization Guide

Interactive hemisphere visualization for analyzing BRDF data.

---

## Features

✅ **Implemented (Phase 1 + Option B)**

- **File Upload** - Load MERL .binary files through web interface
- **Hemisphere Plot** - 3D visualization of reflection distribution
- **Interactive Controls** - Real-time incident angle adjustment
- **Color-coded Display** - Luminance-based coloring
- **Incident Ray Indicator** - Red arrow showing light direction

---

## Quick Start

### 1. Start the Interactive Server

```bash
julia --project=. start_interactive.jl
```

### 2. Open in Browser

Navigate to: `http://127.0.0.1:8000`

### 3. Load BRDF Data

- Click **"Choose File"** button
- Select a MERL `.binary` file
- Wait for processing (1-2 seconds)
- Status will show:
  - ✅ Material name
  - Format (merl)
  - Number of samples (1,458,000)
  - Wavelengths (630, 532, 465 nm)

### 4. Explore the Data

- **Adjust Polar Angle (θ)**: Slider from 0° to 90°
  - 0° = Light from directly above (zenith)
  - 45° = Common viewing angle
  - 90° = Grazing angle (horizon)

- **Adjust Azimuthal Angle (φ)**: Slider from 0° to 360°
  - Changes horizontal direction of incident light

- **Watch the plot update** in real-time!

---

## Understanding the Visualization

### Hemisphere Plot

The 3D plot shows:

- **Surface Shape**: Deformed hemisphere
  - Outward bulges = Strong reflection in that direction
  - Flat areas = Weak reflection
  - Base circle = Horizon plane

- **Colors**: Thermal colormap (blue → red)
  - Blue/Purple = Low reflection
  - Yellow/Orange = Medium reflection
  - Red/White = High reflection (specular peaks)

- **Red Arrow**: Incident light direction
  - Points away from surface (convention)
  - Label: "Incident Light"

### Controls

**Polar Angle (θ)**
- Range: 0° - 90°
- **0°**: Light from directly overhead
  - Symmetric reflection pattern
  - Good for comparing diffuse vs specular
- **45°**: Typical viewing scenario
  - Balanced diffuse and specular
- **90°**: Grazing incidence
  - Strong specular highlight
  - Minimal diffuse component

**Azimuthal Angle (φ)**
- Range: 0° - 360°
- **Effect depends on material:**
  - Isotropic (MERL): No change (rotationally symmetric)
  - Anisotropic: Changes reflection pattern

*Note: MERL materials are isotropic, so φ has limited effect*

---

## Material Appearance Examples

### Diffuse Materials
- **Shape**: Nearly spherical bulge
- **Colors**: Mostly uniform
- **Peak**: At surface normal (straight up)
- **Example**: Matte paint, paper, unpolished clay

### Glossy Materials
- **Shape**: Elongated lobe
- **Colors**: Gradient from dim to bright
- **Peak**: Near mirror direction
- **Example**: Polished wood, satin fabric

### Specular Materials
- **Shape**: Narrow spike
- **Colors**: Concentrated bright spot
- **Peak**: Exactly at mirror direction
- **Example**: Chrome, mirror, polished metal

### Rough Materials
- **Shape**: Broad, uneven distribution
- **Colors**: Scattered patches
- **Peak**: Diffuse with rough specular
- **Example**: Brushed aluminum, fabric, rough plastic

---

## Technical Details

### Coordinate System

- **Z-axis**: Surface normal (pointing up)
- **X-Y plane**: Surface tangent plane
- **Hemisphere**: All directions above surface (z ≥ 0)

### BRDF Evaluation

The visualization uses the Rusinkiewicz half-angle parameterization:

```
θ_h = acos(h · n)       # Half-vector polar angle
θ_d = acos(ωi · h)     # Difference angle
φ_d = atan2(...)       # Difference azimuth
```

Where:
- `h = normalize(ωi + ωo)` - Half-vector
- `ωi` - Incident direction (from slider)
- `ωo` - Outgoing direction (hemisphere grid)
- `n = (0, 0, 1)` - Surface normal

### Color Mapping

```julia
luminance = 0.299*R + 0.587*G + 0.114*B
scale = 1.0 + luminance * 3.0
color = thermal_colormap(luminance)
```

The luminance determines:
1. **Radial scale** - How far point extends from origin
2. **Color** - Mapped through thermal colormap

---

## Performance Notes

### Resolution

Default: **30×30 grid** (900 BRDF evaluations)

- **Lower** (20×20): Faster, blockier
- **Higher** (50×50): Slower, smoother
- **Recommended**: 30×30 for interactive use

### Rendering Speed

- **Initial load**: 1-2 seconds
- **Angle update**: ~0.5 seconds
- **Browser requirements**: WebGL support

### File Size Limits

- MERL files: 17,496,000 bytes (fixed)
- Upload limit: ~20 MB (Bonito default)
- Memory: ~50 MB per loaded file

---

## Getting MERL Data

### Download MERL BRDF Database

```bash
# Example materials (hypothetical URLs - check actual source):
wget http://merl.com/brdf/alum-bronze.binary
wget http://merl.com/brdf/chrome.binary
wget http://merl.com/brdf/gold-paint.binary
```

**Official Source**: [MERL BRDF Database](https://www.merl.com/brdf/)

### Included Test Data

Place test files in:
```
BRDFToolkit/data/merl/
```

---

## Troubleshooting

### "No data loaded" message

**Cause**: File not uploaded or failed to load

**Solution**:
1. Check file extension is `.binary`
2. Verify file size is exactly 17,496,000 bytes
3. Check browser console (F12) for errors
4. Try refreshing page

### Plot appears distorted

**Cause**: Extreme BRDF values or numerical issues

**Solution**:
- Try different incident angles
- Check if material is valid MERL format
- Restart server if visualization freezes

### Sliders don't update plot

**Cause**: WebSocket connection lost

**Solution**:
1. Check server is running
2. Refresh browser page
3. Check browser console for WebSocket errors
4. Restart server

### Slow performance

**Optimization**:
- Close other browser tabs
- Use Chrome/Firefox (best WebGL support)
- Reduce resolution (edit `interactive_app.jl`)
- Check GPU drivers are updated

---

## Advanced Usage

### Custom Resolution

Edit `src/app/interactive_app.jl`:

```julia
resolution = 30  # Change to 20 (faster) or 50 (prettier)
```

### Export Plots

Currently not supported. Coming in Phase 2:
- Save as PNG/SVG
- Export data as CSV
- Generate comparison reports

### Multiple Materials

Currently single-file only. Coming in Phase 2:
- Side-by-side comparison
- Overlay multiple BRDFs
- Material gallery view

---

## Keyboard Shortcuts

*Coming in Phase 2*

Planned:
- `R` - Reset view
- `Space` - Play/pause animation
- `←/→` - Adjust theta
- `↑/↓` - Adjust phi

---

## API Usage (Programmatic)

### Create Plot from Code

```julia
using BRDFPlatform

# Load data
data = load_brdf("alum-bronze.binary")

# Create plot
fig = create_hemisphere_plot(data, Float32(π/4), Float32(0.0))

# Save or display
save("output.png", fig)
```

### Batch Processing

```julia
using BRDFPlatform

for file in readdir("data/merl")
    if endswith(file, ".binary")
        data = load_brdf(joinpath("data/merl", file))
        fig = create_simple_hemisphere_plot(data)
        save("output/$(data.name).png", fig)
    end
end
```

---

## Future Enhancements

### Phase 2 (Planned)

- [ ] Angular slice plots
- [ ] Error heatmaps (fitted vs measured)
- [ ] Wavelength selection
- [ ] Multiple incident angles simultaneously
- [ ] Animation/rotation controls
- [ ] Export functionality

### Phase 3 (Optional)

- [ ] Three.js material preview
- [ ] Real-time model fitting visualization
- [ ] Comparison matrix
- [ ] VR/AR preview

---

## Resources

**Theory**:
- [MERL BRDF Database](https://www.merl.com/brdf/)
- Rusinkiewicz parameterization paper
- BRDF fundamentals tutorials

**Tools**:
- WGLMakie documentation
- Bonito.jl examples
- Julia graphics ecosystem

---

## Support

**Issues**:
- Check `QUICKSTART.md` for common problems
- Review `UI_GUIDE.md` for interface help
- See `README.md` for project overview

**Contact**:
- Open GitHub issue (when repo created)
- Check `CLAUDE.MD` for development notes

---

**Last Updated**: January 30, 2025
**Version**: Phase 1 + Visualization (Option B Complete)
