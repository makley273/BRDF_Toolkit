# BRDF Platform UI Guide

Interactive web interface for the BRDF Analysis Platform.

## Quick Start

### Starting the Server

**Option 1: Using startup script**
```bash
julia --project=. start_app.jl
```

**Option 2: With custom port**
```bash
julia --project=. start_app.jl --port 8080
```

**Option 3: From Julia REPL**
```julia
using BRDFPlatform
server = run_app()  # Default: http://127.0.0.1:8000

# Or with custom settings
server = run_app(port=8080, host="0.0.0.0")
```

### Accessing the Interface

1. Start the server (see above)
2. Open your web browser
3. Navigate to: `http://127.0.0.1:8000`
4. The interface should load automatically

### Stopping the Server

- Press `Ctrl+C` in the terminal/console
- Or close the Julia session

---

## Interface Overview

The UI is organized into several sections:

### 1. Sidebar (Left)

**Quick Actions:**
- 🔄 **Reload User Plugins** - Refresh plugins from `~/.brdf_platform/`
- 📊 **Export Results** - Export fitted models and metrics (Phase 2)
- ℹ️ **Help** - Documentation and guides

**Quick Guide:**
Step-by-step workflow overview for using the platform.

### 2. Main Content Area

#### Header
- Platform title and description
- Quick info about the tool

#### Plugin Panel
Shows currently loaded plugins:
- **Formats**: Available file format loaders
- **Models**: Analytical BRDF models
- **Optimizers**: Fitting algorithms

#### File Upload Panel
Upload BRDF data files:
- Supported formats: `.binary` (MERL), `.bird`, `.json`
- Drag & drop support (coming soon)
- Shows file info after loading:
  - Material name
  - Detected format
  - Number of samples
  - Wavelengths

#### Model Fitting Panel (Placeholder)
Will include (Phase 2):
- Model selection dropdown
- Optimizer selection
- Parameter inputs
- Fit button
- Fitted parameters display
- Error metrics

#### Visualization Panel (Placeholder)
Will display (Phase 2):
- **WGLMakie plots:**
  - Hemisphere plots
  - Error heatmaps
  - Angular slices
- **Three.js rendering:**
  - 3D material preview
  - Real-time rotation
  - Side-by-side comparison

#### Footer
- Project information
- Links to documentation
- GitHub repository

---

## Current Features (Phase 1)

### ✅ Implemented

1. **Plugin Registry Display**
   - Shows all loaded format/model/optimizer plugins
   - Real-time count updates
   - Plugin names listed

2. **File Upload Interface**
   - File input with format filtering
   - Status display (loading, success, error)
   - File information display

3. **Layout and Navigation**
   - Responsive sidebar
   - Clean, modern design
   - Clear visual hierarchy

4. **Quick Actions**
   - Reload user plugins button
   - Help access
   - Export placeholder

### 🚧 Coming in Phase 2

1. **Interactive File Upload**
   - Drag & drop support
   - Progress indicators
   - Multiple file handling

2. **Model Fitting**
   - Model selection UI
   - Parameter controls
   - Real-time fitting
   - Error metrics display

3. **Scientific Visualization**
   - WGLMakie hemisphere plots
   - Interactive error heatmaps
   - Angular slice views
   - Wavelength selection

4. **Material Rendering**
   - Three.js integration
   - Real-time BRDF preview
   - Interactive lighting
   - Model comparison view

5. **Export Functionality**
   - Save fitted parameters
   - Export visualizations
   - Generate reports

---

## Design Philosophy

### Visual Design

- **Color Scheme:**
  - Primary: #3498db (blue)
  - Success: #27ae60 (green)
  - Warning: #e67e22 (orange)
  - Error: #e74c3c (red)
  - Neutral: #7f8c8d (gray)

- **Typography:**
  - System font stack for cross-platform consistency
  - Clear hierarchy with size and weight
  - Good contrast for readability

- **Layout:**
  - Fixed sidebar for persistent navigation
  - Flexible main content area
  - Two-column layout for comparisons
  - Responsive spacing and padding

### User Experience

- **Progressive Disclosure:**
  - Show essentials first
  - Advanced features appear when needed
  - Clear visual feedback for all actions

- **Error Handling:**
  - Graceful degradation
  - Clear error messages
  - Suggested recovery actions

- **Performance:**
  - Minimal Julia ↔ JavaScript communication
  - Observables for reactive updates
  - Efficient data transfer

---

## Technical Details

### Technology Stack

- **Bonito.jl**: Web framework
  - Session management
  - Observable-based reactivity
  - DOM manipulation
  - WebSocket communication

- **WGLMakie**: Scientific visualization (Phase 2)
  - WebGL-based plotting
  - Interactive 3D plots
  - GPU-accelerated rendering

- **Three.js**: Material rendering (Phase 2)
  - Custom GLSL shaders
  - Real-time lighting
  - Camera controls

### Architecture

```
┌─────────────────────────────────────────┐
│         Browser (Client)                │
│  ┌────────────────────────────────────┐ │
│  │  DOM (Bonito-generated HTML)       │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │  WGLMakie (WebGL canvas)          │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │  Three.js (WebGL canvas)          │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
                   ↕ WebSocket
┌─────────────────────────────────────────┐
│         Julia Server                    │
│  ┌────────────────────────────────────┐ │
│  │  Bonito.jl Server                  │ │
│  │  └─ Session Management             │ │
│  │  └─ Observable Updates             │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │  BRDFPlatform Core                 │ │
│  │  └─ Format Loaders                 │ │
│  │  └─ BRDF Models                    │ │
│  │  └─ Optimizers                     │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

### Data Flow

1. **File Upload:**
   ```
   Browser File Input
      → Julia File Handler
      → Format Detection
      → MeasuredBRDF
      → Observable Update
      → UI Status Display
   ```

2. **Model Fitting (Phase 2):**
   ```
   User Selects Model/Optimizer
      → Julia Fitting Algorithm
      → Fitted Parameters
      → Observable Update
      → UI Display + Visualization
   ```

3. **Visualization (Phase 2):**
   ```
   Fitted Model
      → GLSL Shader Generation
      → Three.js Scene Setup
      → Browser WebGL Rendering
   ```

---

## Customization

### Adding Custom Styles

Edit `src/app/main_app.jl` and modify inline styles or add CSS:

```julia
custom_style = """
<style>
.brdf-panel {
    background: #f0f0f0;
    padding: 20px;
}
</style>
"""
```

### Modifying Layout

The layout is defined in `create_main_app()`:
- Sidebar: `create_sidebar()`
- Header: `create_header()`
- Content panels: Individual component functions

### Adding New Panels

1. Create a new component function:
```julia
function create_my_panel(session)
    return DOM.div(
        DOM.h3("My Custom Panel"),
        DOM.p("Content here"),
        style="padding: 15px;"
    )
end
```

2. Add to main layout in `create_main_app()`:
```julia
create_my_panel(session),
```

---

## Troubleshooting

### Server Won't Start

**Error: "Address already in use"**
- Solution: Change port with `--port 8080`
- Or: Kill process using the port

**Error: "Cannot load Bonito"**
- Solution: Reinstall dependencies:
  ```julia
  ] instantiate
  ```

### UI Not Loading

**Blank page in browser**
- Check console for errors (F12)
- Verify server is running
- Try a different browser

**WebSocket connection failed**
- Check firewall settings
- Verify port is open
- Try `host="0.0.0.0"` for network access

### Performance Issues

**Slow rendering**
- Reduce visualization complexity (Phase 2)
- Check browser GPU acceleration
- Close other tabs

**High memory usage**
- Large BRDF files consume memory
- Restart server periodically
- Consider data subsampling

---

## Development

### Testing UI Changes

```bash
# Run UI tests
julia --project=. test/test_ui.jl

# Run full test suite
julia --project=. test/runtests.jl
```

### Hot Reloading

After modifying `src/app/main_app.jl`:
1. Stop the server (Ctrl+C)
2. Restart Julia session
3. Run `using BRDFPlatform; run_app()`

Or use Revise.jl (recommended for development):
```julia
using Revise
using BRDFPlatform
run_app()
# Now edits to source files reload automatically
```

### Debugging

Enable debug logging:
```julia
ENV["JULIA_DEBUG"] = "Bonito"
using BRDFPlatform
run_app()
```

Check browser console (F12) for JavaScript errors.

---

## Security Notes

### Local Development

- Default: `127.0.0.1` (localhost only)
- No external access
- Safe for local development

### Network Access

If using `host="0.0.0.0"`:
- ⚠️ Accessible from network
- Use firewall rules
- Consider authentication (future)
- Only for trusted networks

### File Uploads

- Validates file extensions
- Checks file size limits
- Sandboxed Julia execution
- No arbitrary code execution

---

## Future Enhancements

### Planned Features

- [ ] User authentication
- [ ] Session persistence
- [ ] Batch processing
- [ ] Cloud deployment support
- [ ] Mobile-responsive design
- [ ] Dark mode theme
- [ ] Keyboard shortcuts
- [ ] Undo/redo for parameters
- [ ] Model comparison matrix
- [ ] Export to multiple formats

### Plugin UI

Future plugin system will support:
- Custom UI panels
- Plugin-specific visualizations
- Configuration interfaces
- Status indicators

---

## Resources

- **Bonito.jl Docs**: https://github.com/SimonDanisch/Bonito.jl
- **WGLMakie Docs**: http://makie.juliaplots.org/stable/
- **Three.js Docs**: https://threejs.org/docs/

## Support

For issues or questions:
1. Check this guide
2. See `README.md` and `CLAUDE.MD`
3. Review `Architecture_Design.md`
4. Open GitHub issue (when repo is created)

---

**Last Updated:** January 30, 2025
**Version:** Phase 1 (Foundation)
