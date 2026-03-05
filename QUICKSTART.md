# Quick Start Guide

Get up and running with the BRDF Analysis Platform in 3 steps.

## Step 1: Install Dependencies

```bash
cd BRDFToolkit
julia --project=.
```

In Julia REPL:
```julia
] instantiate
```

Wait for all packages to install (~325 packages, first time only).

## Step 2: Run Tests (Optional but Recommended)

```bash
julia --project=. test/runtests.jl
```

Expected output:
```
Test Summary:      | Pass  Total
BRDFPlatform Tests |   24     24
```

## Step 3: Start the Web Interface

```bash
julia --project=. start_app.jl
```

You should see:
```
🎨 Welcome to BRDF Analysis Platform
📚 For help, see: README.md or CLAUDE.MD

======================================================================
🚀 Starting BRDF Analysis Platform
======================================================================
📡 Server running at: http://127.0.0.1:8000
👉 Open your browser and navigate to the URL above
⏹️  Press Ctrl+C to stop the server
======================================================================

✨ Server is running. Waiting for connections...
```

## Step 4: Open in Browser

Open your web browser and navigate to:
```
http://127.0.0.1:8000
```

You should see the BRDF Analysis Platform interface with:
- 📦 Plugin panel (showing 1 format loaded)
- 📂 File upload interface
- 🎯 Model fitting panel (placeholder)
- 📊 Visualization panel (placeholder)

## Common Issues

### "Address already in use"
Another program is using port 8000. Use a different port:
```bash
julia --project=. start_app.jl --port 8080
```

### "Package not found"
Run `] instantiate` again in Julia REPL.

### Browser shows blank page
- Check if server is running (look for "Server running" message)
- Try refreshing the page (Ctrl+R or F5)
- Check browser console for errors (F12)

## What You Can Do Now

**Phase 1 Features (Current):**
- ✅ View loaded plugins
- ✅ See the web interface
- ✅ Explore the UI layout

**Coming in Phase 2:**
- Upload BRDF data files
- Fit analytical models
- Visualize results
- Compare models

## Next Steps

### For Users:
- See `UI_GUIDE.md` for detailed interface documentation
- See `README.md` for project overview

### For Developers:
- See `CLAUDE.MD` for comprehensive development guide
- See `Architecture_Design.md` for system architecture
- See `TODO.md` for implementation roadmap

## Getting Help

1. Check the documentation files listed above
2. Review `PHASE1_COMPLETE.md` for current status
3. Open an issue on GitHub (when repo is created)

---

**Last Updated:** January 30, 2025
