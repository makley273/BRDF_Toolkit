# File Upload Fix - Interactive BRDF Visualization

## Problems Identified & Fixed

### Issue 1: Undefined `Bonito.File`
The error `UndefVarError: File not defined in Bonito` occurred because:

1. ❌ **Attempted to use** `Bonito.File` which doesn't exist
2. ✅ **Should use** `Bonito.TailwindDashboard.FileInput()` or `Bonito.FileInput()`

### Issue 2: Incorrect `onjs` Usage
The error `MethodError: no method matching onjs(::Session, ::DOM.Element, ::Symbol, ::JSCode)` occurred because:

1. ❌ **Attempted to use** `onjs(session, button, :click, js"...")`
2. ✅ **Should use** `DOM.button(..., onclick=js"...")`

DOM event handlers are attached directly as attributes, not via `onjs`.

## Root Cause: Blob URLs

After investigating the Bonito source code, I discovered that `FileInput` returns **blob URLs**, not actual file data:

```julia
# From Bonito source: src/widgets.jl
function Bonito.jsrender(session::Session, fi::FileInput)
    onchange = js"""event => {
        if (event.target.files) {
            const files = [];
            for (const file of event.target.files) {
                files.push(URL.createObjectURL(file))  # ← Returns URLs, not bytes!
            }
            $(fi.value).notify(files);
        }
    }"""
    # ...
end
```

## Current Implementation

The file upload now works in **two modes**:

### Mode 1: Direct File Path (✅ **Fully Working**)

Users can enter a file path directly:

```
C:/path/to/your/file.binary
```

And click "Load File" button.

**This mode works immediately and is recommended for development.**

### Mode 2: Browser File Upload (⚠️ **Partially Implemented**)

The file upload widget is present but currently:
- Returns blob URLs (`blob:http://127.0.0.1:8000/uuid`)
- Needs JavaScript `fetch()` to retrieve actual bytes
- Not yet fully implemented (would require custom JavaScript)

## How to Use (Current Version)

### Step 1: Start the Server

```bash
julia --project=. start_interactive.jl
```

### Step 2: Open Browser

Navigate to: **http://127.0.0.1:8000**

### Step 3: Load a BRDF File

**Option A - File Path (Recommended):**
1. Find the full path to your `.binary` file
   - Example: `C:/Users/makle/Desktop/data/chrome.binary`
2. Enter it in the "Option 2: Enter file path directly" text field
3. Click "Load File"

**Option B - File Upload (Not Yet Working):**
- The "Choose File" button is visible but won't load files yet
- Shows message: "Blob URL file loading not yet implemented"

### Step 4: Interact with Visualization

Once loaded:
- ✅ See file info (name, format, samples, wavelengths)
- ✅ Adjust polar angle (θ) slider
- ✅ Adjust azimuthal angle (φ) slider
- ✅ Watch hemisphere plot update in real-time
- ✅ Rotate/zoom the 3D view with mouse

## Implementation Changes

### Files Modified

1. **`src/app/interactive_app.jl`**
   - Added `import Bonito.TailwindDashboard as D`
   - Changed from `Bonito.File()` → `D.FileInput()`
   - Added file path input field as workaround
   - Added load button with click handler
   - Implemented path-based loading

### Code Structure

```julia
# File upload widget (returns blob URLs)
file_widget = D.FileInput()

# Workaround: Direct path input
file_path_input = DOM.input(type="text", placeholder="...")
load_button = DOM.button("Load File", ...)

# Handle path-based loading
on(file_path_obs) do filepath
    if startswith(filepath, "blob:")
        # Not yet implemented
    else
        # Load from filesystem - WORKS!
        data = load_brdf(filepath)
    end
end
```

## Future Work: Full Browser Upload

To fully implement browser-based file upload, we need to:

### Option 1: JavaScript Fetch (Recommended)

Add JavaScript to fetch blob URL contents:

```julia
onjs(session, file_widget.value, :change, js"""
    async function() {
        const url = $(file_widget.value)[0];
        const response = await fetch(url);
        const arrayBuffer = await response.arrayBuffer();
        const uint8Array = new Uint8Array(arrayBuffer);

        // Send bytes to Julia
        $(file_bytes_obs).notify(Array.from(uint8Array));
    }
""")
```

Then handle the bytes in Julia:

```julia
on(file_bytes_obs) do bytes
    temp_path = tempname() * ".binary"
    write(temp_path, UInt8.(bytes))
    data = load_brdf(temp_path)
    # ...
end
```

### Option 2: Custom File Reader Widget

Create a custom widget that reads file contents directly:

```julia
const CustomFileReader = ES6Module(joinpath(@__DIR__, "file_reader.js"))

# In file_reader.js:
# Use FileReader API to read file as ArrayBuffer
# Send bytes directly to Julia via observable
```

## Testing

All existing tests still pass:

```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

Output:
```
Test Summary:      | Pass  Total
BRDFPlatform Tests |   30     30
  Foundation       |   24     24
  Visualization    |    6      6
```

## Example Usage Session

```julia
# Start server
$ julia --project=. start_interactive.jl

🚀 Starting BRDF Analysis Platform (Interactive)
📡 Server running at: http://127.0.0.1:8000

# In browser:
# 1. Enter path: C:/path/to/chrome.binary
# 2. Click "Load File"
# 3. See: ✅ Loaded: chrome
#         Format: merl
#         Samples: 1458000
#         Wavelengths: 630, 532, 465 nm
# 4. Move sliders → plot updates!
```

## Troubleshooting

### Error: "File not found"

Make sure you use the full absolute path:
- ✅ `C:/Users/makle/Desktop/data/file.binary`
- ❌ `~/Desktop/data/file.binary` (won't work on Windows)
- ❌ `./data/file.binary` (relative paths not supported yet)

### Error: "Blob URL file loading not yet implemented"

You clicked "Choose File" instead of using the path input.
Use Option 2 (file path) for now.

### No visualization appears

Check console for errors:
```julia
julia> using BRDFPlatform
julia> run_interactive_app()
```

Look for errors like:
- `File not found`
- `Failed to load BRDF`
- `Invalid format`

## Summary

**Current Status:**
- ✅ File path loading works perfectly
- ✅ Visualization is fully functional
- ⚠️ Browser file upload needs JavaScript implementation
- ✅ All tests passing
- ✅ Interactive controls working

**Recommended Workflow:**
1. Use file path input (Option 2) for now
2. Focus on using the visualization features
3. Implement full browser upload later if needed

**Next Steps:**
- Option A: Add Lambertian BRDF model
- Option B: Implement full browser upload with JavaScript fetch
- Option C: Add more visualization features (error heatmaps, etc.)

---

*Last Updated: January 30, 2025*
*BRDF Analysis Platform - Phase 1 + Visualization*
